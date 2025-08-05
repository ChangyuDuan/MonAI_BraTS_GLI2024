import os
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
from monai.networks.nets import UNet, SegResNet, UNETR, SwinUNETR, AttentionUnet, VNet, HighResNet
from monai.losses import DiceCELoss, FocalLoss, TverskyLoss, GeneralizedDiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, ConfusionMatrixMetric, MeanIoU, GeneralizedDiceScore
from monai.inferers import sliding_window_inference

from knowledge_distillation import (
    KnowledgeDistillationLoss, 
    MultiTeacherDistillation, 
    ProgressiveKnowledgeDistillation
)
from fusion_architectures import FusionNetworkArchitecture
from nas_search import (
    SuperNet,
    DARTSSearcher,
    ProgressiveNAS
)


class BasicModelBank:
    """
    简化的分割模型，支持基础模型架构和多种损失函数策略
    """
    def __init__(self, model_name: str = 'UNet', device: str = 'auto'):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = self._create_model()
        self.loss_function = self._create_adaptive_loss()
        self.metrics = self._create_metrics()
        
    def _setup_device(self, device) -> torch.device:
        """设置设备"""
        # 如果已经是torch.device对象，直接返回
        if isinstance(device, torch.device):
            return device
        
        # 处理字符串类型的设备参数
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif str(device).lower() == 'cpu':
            return torch.device('cpu')
        elif str(device).lower() == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("警告: CUDA不可用，自动切换到CPU")
                return torch.device('cpu')
        else:
            return torch.device(device)
        
    def _create_model(self):
        """创建模型"""
        if self.model_name == 'UNet':
            model = UNet(
                spatial_dims=3,
                in_channels=4,
                out_channels=4,
                channels=(32, 64, 128, 256, 512),
                strides=(2, 2, 2, 2),
                num_res_units=2,
                norm="instance",
                dropout=0.1
            )
        elif self.model_name == 'SegResNet':
            model = SegResNet(
                spatial_dims=3,
                init_filters=32,
                in_channels=4,
                out_channels=4,
                dropout_prob=0.1,
                act=("RELU", {"inplace": True}),
                norm=("GROUP", {"num_groups": 8}),
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                upsample_mode="nontrainable"
            )
        elif self.model_name == 'UNETR':
            model = UNETR(
                in_channels=4,
                out_channels=4,
                img_size=(128, 128, 128),
                feature_size=16,
                hidden_size=768,
                mlp_dim=3072,
                num_heads=12,
                proj_type="conv",
                norm_name="instance",
                conv_block=True,
                res_block=True,
                dropout_rate=0.1,
                spatial_dims=3,
                qkv_bias=False,
                save_attn=False
            )
        elif self.model_name == 'SwinUNETR':
            model = SwinUNETR(
                in_channels=4,
                out_channels=4,
                feature_size=48,
                depths=(2, 2, 2, 2),
                num_heads=(3, 6, 12, 24),
                norm_name="instance",
                drop_rate=0.0,
                attn_drop_rate=0.0,
                dropout_path_rate=0.0,
                normalize=True,
                use_checkpoint=False,
                spatial_dims=3,
                downsample="merging",
                use_v2=False
            )
        elif self.model_name == 'AttentionUNet':
            model = AttentionUnet(
                spatial_dims=3,
                in_channels=4,
                out_channels=4,
                channels=(32, 64, 128, 256),
                strides=(2, 2, 2),
                dropout=0.1
            )
        elif self.model_name == 'VNet':
            model = VNet(
                spatial_dims=3,
                in_channels=4,
                out_channels=4,
                act=("elu", {"inplace": True}),
                dropout_prob_down=0.5,
                dropout_prob_up=(0.5, 0.5),
                dropout_dim=3,
                bias=False
            )
        elif self.model_name == 'HighResNet':
            model = HighResNet(
                spatial_dims=3,
                in_channels=4,
                out_channels=4,
                norm_type=("batch", {"affine": True}),
                acti_type=("relu", {"inplace": True}),
                dropout_prob=0.1,
                bias=False,
                layer_params=(
                    {"name": "conv_0", "n_features": 16, "kernel_size": 3},
                    {"name": "res_1", "n_features": 16, "kernels": (3, 3), "repeat": 3},
                    {"name": "res_2", "n_features": 32, "kernels": (3, 3), "repeat": 3},
                    {"name": "res_3", "n_features": 64, "kernels": (3, 3), "repeat": 3},
                    {"name": "conv_1", "n_features": 80, "kernel_size": 1},
                    {"name": "conv_2", "kernel_size": 1},
                ),
                channel_matching="pad"
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_name}")
            
        return model.to(self.device)
    
    def _create_adaptive_loss(self):
        """创建自适应损失函数调度器"""
        class AdaptiveLossScheduler:
            def __init__(self, parent):
                self.parent = parent
                self.losses = {
                    'dice_ce': DiceCELoss(to_onehot_y=True, softmax=True, reduction="mean", include_background=True),
                    'focal': FocalLoss(to_onehot_y=True, gamma=2.0, reduction="mean", include_background=True),
                    'tversky': TverskyLoss(to_onehot_y=True, softmax=True, alpha=0.3, beta=0.7, reduction="mean", include_background=True),
                    'generalized_dice': GeneralizedDiceLoss(to_onehot_y=True, softmax=True, reduction="mean", include_background=True),
                    'dice_focal': DiceFocalLoss(to_onehot_y=True, softmax=True, gamma=2.0, reduction="mean", include_background=True)
                }
                self.current_epoch = 0
                self.total_epochs = 100
                
            def set_epoch(self, epoch, total_epochs=None):
                self.current_epoch = epoch
                if total_epochs is not None:
                    self.total_epochs = total_epochs
                    
            def _calculate_adaptive_weights(self, progress):
                """根据训练进度自适应调整权重"""
                if progress < 0.2:  # 前20%: 主要使用DiceCE
                    return {'dice_ce': 0.7, 'focal': 0.2, 'tversky': 0.1, 'generalized_dice': 0.0, 'dice_focal': 0.0}
                elif progress < 0.4:  # 20%-40%: 增加Focal权重
                    return {'dice_ce': 0.5, 'focal': 0.3, 'tversky': 0.1, 'generalized_dice': 0.1, 'dice_focal': 0.0}
                elif progress < 0.6:  # 40%-60%: 平衡各损失
                    return {'dice_ce': 0.3, 'focal': 0.3, 'tversky': 0.2, 'generalized_dice': 0.1, 'dice_focal': 0.1}
                elif progress < 0.8:  # 60%-80%: 增加Tversky权重
                    return {'dice_ce': 0.2, 'focal': 0.2, 'tversky': 0.4, 'generalized_dice': 0.1, 'dice_focal': 0.1}
                else:  # 最后20%: 组合所有损失
                    return {'dice_ce': 0.2, 'focal': 0.2, 'tversky': 0.2, 'generalized_dice': 0.2, 'dice_focal': 0.2}
                    
            def __call__(self, pred, target):
                progress = self.current_epoch / self.total_epochs if self.total_epochs > 0 else 0
                weights = self._calculate_adaptive_weights(progress)
                
                total_loss = 0
                for name, weight in weights.items():
                    if weight > 0:
                        total_loss += weight * self.losses[name](pred, target)
                        
                return total_loss
                    
        return AdaptiveLossScheduler(self)
        
    def _create_metrics(self):
        """创建评估指标"""
        metrics = {
            # 基础指标
            'dice': DiceMetric(
                include_background=False,
                reduction="mean_batch",
                get_not_nans=False
            ),
            'hausdorff': HausdorffDistanceMetric(
                include_background=False,
                distance_metric='euclidean',
                percentile=95,
                directed=False,
                reduction="mean_batch"
            ),
            # 高级指标
            'surface_distance': SurfaceDistanceMetric(
                include_background=False,
                symmetric=True,
                reduction="mean_batch"
            ),
            'confusion_matrix': ConfusionMatrixMetric(
                include_background=False,
                metric_name="sensitivity",
                compute_sample=True,
                reduction="mean_batch"
            ),
            'mean_iou': MeanIoU(
                include_background=False,
                reduction="mean_batch"
            ),
            'generalized_dice_score': GeneralizedDiceScore(
                include_background=False,
                reduction="mean_batch"
            )
        }
        
        return metrics
        
    def get_model(self):
        """获取模型"""
        return self.model
        
    def get_loss_function(self):
        """获取损失函数"""
        return self.loss_function
        
    def get_metrics(self):
        """获取评估指标"""
        return self.metrics
    
    def update_loss_epoch(self, epoch: int, total_epochs: int = None):
        """更新自适应损失函数的epoch信息"""
        if hasattr(self.loss_function, 'set_epoch'):
            self.loss_function.set_epoch(epoch, total_epochs)
    
    def get_loss_info(self) -> Dict[str, Any]:
        """获取损失函数信息"""
        info = {
            'strategy': 'adaptive_combined',
            'type': 'adaptive_multiple',
            'description': 'Adaptive combination of all loss functions with dynamic weights'
        }
        return info
        
    def sliding_window_inference(self, inputs: torch.Tensor, 
                                roi_size: Tuple = (128, 128, 128),
                                sw_batch_size: int = 4,
                                overlap: float = 0.6) -> torch.Tensor:
        """滑动窗口推理"""
        return sliding_window_inference(
            inputs=inputs,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=self.model,
            overlap=overlap,
            mode="gaussian",
            sw_device=self.device,
            device=self.device
        )


def create_optimizer(model: nn.Module, 
                    optimizer_name: str = "AdamW",
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-5) -> optim.Optimizer:
    """创建优化器"""
    if optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
    elif optimizer_name.lower() == "sgd":
        return optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9
        )
    else:
        return optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )


def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_name: str = "CosineAnnealingLR",
                    **kwargs):
    """创建学习率调度器"""
    if scheduler_name.lower() == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
    elif scheduler_name.lower() == "cosineannealinglr":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100)
        )
    elif scheduler_name.lower() == "reducelronplateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10)
        )
    else:
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 100)
        )

def get_all_supported_models() -> List[str]:
    """
    获取所有支持的模型列表
    """
    return ['UNet', 'SegResNet', 'UNETR', 'SwinUNETR', 'AttentionUNet', 'VNet', 'HighResNet']


class SpecializedModelFactory:
    """
    知识蒸馏、融合网络和神经架构搜索
    """
    
    def __init__(self, 
                 model_type: str = 'fusion',  # 'fusion', 'distillation', 'nas'
                 device: str = 'auto',
                 **kwargs):
        self.model_type = model_type
        self.device = self._setup_device(device)
        self.kwargs = kwargs
        self.model = self._create_advanced_model()
        
    def _setup_device(self, device) -> torch.device:
        """设置设备"""
        if isinstance(device, torch.device):
            return device
        
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif str(device).lower() == 'cpu':
            return torch.device('cpu')
        elif str(device).lower() == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("警告: CUDA不可用，自动切换到CPU")
                return torch.device('cpu')
        else:
            return torch.device(device)
            
    def _create_advanced_model(self):
        """创建高级模型"""
        if self.model_type == 'fusion':
            return self._create_fusion_model()
        elif self.model_type == 'distillation':
            return self._create_distillation_model()
        elif self.model_type == 'nas':
            return self._create_nas_model()
        else:
            raise ValueError(f"不支持的高级模型类型: {self.model_type}")
            
    def _create_fusion_model(self):
        """创建融合网络模型"""
        # 获取基础模型列表，默认使用所有7个网络架构
        base_model_names = self.kwargs.get('base_models', get_all_supported_models())
        base_models = []
        
        for model_name in base_model_names:
            basic_model = BasicModelBank(model_name=model_name, device=self.device)
            base_models.append(basic_model.model)
            
        # 创建融合网络
        fusion_model = FusionNetworkArchitecture(
            base_models=base_models,
            fusion_channels=self.kwargs.get('fusion_channels', [64, 128, 256, 512]),
            num_classes=self.kwargs.get('num_classes', 4),
            fusion_type=self.kwargs.get('fusion_type', 'cross_attention')
        )
        
        return fusion_model.to(self.device)
        
    def _create_distillation_model(self):
        """创建知识蒸馏模型"""
        distillation_type = self.kwargs.get('distillation_type', 'multi_teacher')
        
        if distillation_type == 'multi_teacher':
            # 创建教师模型，默认使用所有7个网络架构作为教师模型
            teacher_names = self.kwargs.get('teacher_models', get_all_supported_models())
            teacher_models = []
            
            for teacher_name in teacher_names:
                teacher = BasicModelBank(model_name=teacher_name, device=self.device)
                
                # 尝试加载预训练权重
                pretrained_dir = self.kwargs.get('pretrained_dir', './pretrained_teachers')
                pretrained_path = f"{pretrained_dir}/{teacher_name}_pretrained.pth"
                
                if os.path.exists(pretrained_path):
                    try:
                        print(f"加载教师模型 {teacher_name} 的预训练权重: {pretrained_path}")
                        checkpoint = torch.load(pretrained_path, map_location=self.device)
                        
                        # 尝试不同的权重键名
                        if 'model_state_dict' in checkpoint:
                            teacher.model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            teacher.model.load_state_dict(checkpoint['state_dict'])
                        else:
                            teacher.model.load_state_dict(checkpoint)
                        
                        print(f"✓ 成功加载教师模型 {teacher_name} 的预训练权重")
                        
                        # 获取预训练性能信息
                        if 'best_metric' in checkpoint:
                            print(f"  预训练最佳指标: {checkpoint['best_metric']:.4f}")
                            
                    except Exception as e:
                        print(f"⚠ 教师模型 {teacher_name} 预训练权重加载失败: {e}")
                        print(f"  将使用随机初始化权重")
                else:
                    print(f"⚠ 未找到教师模型 {teacher_name} 的预训练权重: {pretrained_path}")
                    print(f"  将使用随机初始化权重")
                
                teacher_models.append(teacher.model)
                
            # 创建学生模型
            student_name = self.kwargs.get('student_model', 'UNet')
            student = BasicModelBank(model_name=student_name, device=self.device)
            
            # 创建多教师蒸馏模型
            distillation_model = MultiTeacherDistillation(
                teacher_models=teacher_models,
                student_model=student.model,
                device=self.device,
                temperature=self.kwargs.get('temperature', 4.0)
            )
            
        elif distillation_type == 'progressive':
            # 创建渐进式蒸馏模型
            teacher_name = self.kwargs.get('teacher_model', 'UNETR')
            student_name = self.kwargs.get('student_model', 'UNet')
            
            teacher = BasicModelBank(model_name=teacher_name, device=self.device)
            student = BasicModelBank(model_name=student_name, device=self.device)
            
            distillation_model = ProgressiveKnowledgeDistillation(
                teacher_model=teacher.model,
                student_model=student.model,
                num_stages=self.kwargs.get('num_stages', 3),
                temperature_schedule=self.kwargs.get('temperature_schedule', [6.0, 4.0, 2.0])
            )
            
        else:
            raise ValueError(f"不支持的蒸馏类型: {distillation_type}")
            
        return distillation_model.to(self.device)
        
    def _create_nas_model(self):
        """创建神经架构搜索模型"""
        nas_type = self.kwargs.get('nas_type', 'supernet')
        
        if nas_type == 'supernet':
            # 创建超网络
            supernet = SuperNet(
                in_channels=self.kwargs.get('in_channels', 4),
                num_classes=self.kwargs.get('num_classes', 4),
                base_channels=self.kwargs.get('base_channels', 32),
                num_layers=self.kwargs.get('num_layers', 4)
            )
            return supernet.to(self.device)
            
        elif nas_type == 'searcher':
            # 创建DARTS搜索器
            supernet = SuperNet(
                in_channels=self.kwargs.get('in_channels', 4),
                num_classes=self.kwargs.get('num_classes', 4),
                base_channels=self.kwargs.get('base_channels', 32),
                num_layers=self.kwargs.get('num_layers', 4)
            )
            
            searcher = DARTSSearcher(
                supernet=supernet,
                device=self.device,
                arch_lr=self.kwargs.get('arch_lr', 3e-4),
                model_lr=self.kwargs.get('model_lr', 1e-3)
            )
            return searcher
            
        elif nas_type == 'progressive':
            # 创建渐进式NAS
            progressive_nas = ProgressiveNAS(
                device=self.device,
                max_layers=self.kwargs.get('max_layers', 8),
                start_layers=self.kwargs.get('start_layers', 2)
            )
            return progressive_nas
            
        else:
            raise ValueError(f"不支持的NAS类型: {nas_type}")
            
    def get_model(self):
        """获取模型"""
        return self.model
        
    def get_loss_function(self):
        """获取损失函数"""
        if self.model_type == 'distillation':
            return KnowledgeDistillationLoss(
                temperature=self.kwargs.get('temperature', 4.0),
                alpha=self.kwargs.get('alpha', 0.7)
            )
        else:
            # 使用标准损失函数
            return DiceCELoss(to_onehot_y=True, softmax=True)
            
    def train_step(self, data, labels, optimizer=None):
        """训练步骤"""
        if self.model_type == 'nas' and hasattr(self.model, 'search_step'):
            # NAS搜索步骤
            if len(data) >= 2 and len(labels) >= 2:
                train_data, val_data = data[:len(data)//2], data[len(data)//2:]
                train_labels, val_labels = labels[:len(labels)//2], labels[len(labels)//2:]
                return self.model.search_step(train_data, train_labels, val_data, val_labels)
            else:
                raise ValueError("NAS搜索需要训练和验证数据")
        else:
            # 标准训练步骤
            if optimizer is None:
                raise ValueError("标准训练需要优化器")
                
            optimizer.zero_grad()
            
            if self.model_type == 'distillation':
                # 知识蒸馏训练
                student_output, loss = self.model(data, labels)
                # 损失已经在模型内部计算完成
            else:
                # 标准训练
                output = self.model(data)
                loss_fn = self.get_loss_function()
                loss = loss_fn(output, labels)
                
            loss.backward()
            optimizer.step()
            
            return loss.item()


class ModelFactory:
    """
    模型工厂：统一创建各种类型的模型
    """
    
    @staticmethod
    def create_model(model_config: Dict[str, Any]) -> Any:
        """
        根据配置创建模型
        """
        model_category = model_config.get('category', 'basic')  # 'basic', 'advanced'
        
        if model_category == 'basic':
            return BasicModelBank(
                model_name=model_config.get('model_name', 'UNet'),
                device=model_config.get('device', 'auto')
            )
            
        elif model_category == 'advanced':
            return SpecializedModelFactory(
                model_type=model_config.get('model_type', 'fusion'),
                device=model_config.get('device', 'auto'),
                **model_config.get('kwargs', {})
            )
            
        else:
            raise ValueError(f"不支持的模型类别: {model_category}")
            
    @staticmethod
    def get_model_info() -> Dict[str, List[str]]:
        """
        获取所有支持的模型信息
        """
        return {
            'basic_models': get_all_supported_models(),
            'advanced_types': ['fusion', 'distillation', 'nas'],
            'fusion_types': ['cross_attention', 'channel_attention', 'spatial_attention', 'adaptive'],
            'distillation_types': ['multi_teacher', 'progressive'],
            'nas_types': ['supernet', 'searcher', 'progressive']
        }


if __name__ == "__main__":
    # 功能验证
    print("模型库功能验证...")
    
    # 显示支持的模型列表
    supported_models = get_all_supported_models()
    print(f"基础模型数量: {len(supported_models)}")
    print(f"基础模型列表: {supported_models}")
    
    # 显示高级模型信息
    model_info = ModelFactory.get_model_info()
    print(f"\n高级模型信息:")
    for category, models in model_info.items():
        print(f"  {category}: {models}")
    
    # 创建测试模型验证设备配置
    test_model = BasicModelBank()
    device = test_model._setup_device('auto')
    print(f"\n设备配置验证通过，设备: {device}")
    
    # 测试模型工厂
    try:
        # 测试基础模型创建
        basic_config = {
            'category': 'basic',
            'model_name': 'UNet',
            'device': 'auto'
        }
        basic_model = ModelFactory.create_model(basic_config)
        print(f"基础模型创建成功: {type(basic_model).__name__}")
        
        # 测试高级模型创建（融合网络）
        fusion_config = {
            'category': 'advanced',
            'model_type': 'fusion',
            'device': 'auto',
            'kwargs': {
                'base_models': ['UNet', 'SegResNet'],
                'fusion_type': 'cross_attention',
                'num_classes': 4
            }
        }
        fusion_model = ModelFactory.create_model(fusion_config)
        print(f"融合模型创建成功: {type(fusion_model).__name__}")
        
        print("\n所有功能验证通过！")
        
    except Exception as e:
        print(f"\n功能验证失败: {e}")
        print("注意: 某些功能可能需要额外的依赖或数据")