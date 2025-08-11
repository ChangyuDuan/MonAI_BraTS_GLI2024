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
    支持多数据集动态通道配置
    """
    def __init__(self, model_name: str = 'UNet', device: str = 'auto', dataset_type: str = 'BraTS'):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.dataset_type = dataset_type
        self.in_channels, self.out_channels = self._get_dataset_channels(dataset_type)
        self.model = self._create_model()
        self.loss_function = self._create_adaptive_loss()
        self.metrics = self._create_metrics()
        
    def _get_dataset_channels(self, dataset_type: str) -> Tuple[int, int]:
        """根据数据集类型获取输入输出通道数"""
        dataset_configs = {
            'BraTS': {'in_channels': 4, 'out_channels': 4},  # T1, T1ce, T2, FLAIR -> ET, TC, WT, Background
            'MS_MultiSpine': {'in_channels': 2, 'out_channels': 6}  # T2, STIR/PSIR/MP2RAGE -> 6 lesion classes
        }
        
        if dataset_type not in dataset_configs:
            print(f"警告: 未知数据集类型 '{dataset_type}'，使用BraTS默认配置")
            dataset_type = 'BraTS'
            
        config = dataset_configs[dataset_type]
        print(f"数据集配置 - {dataset_type}: 输入通道={config['in_channels']}, 输出通道={config['out_channels']}")
        return config['in_channels'], config['out_channels']
        
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
                in_channels=self.in_channels,
                out_channels=self.out_channels,
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
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                dropout_prob=0.1,
                act=("RELU", {"inplace": True}),
                norm=("GROUP", {"num_groups": 8}),
                blocks_down=(1, 2, 2, 4),
                blocks_up=(1, 1, 1),
                upsample_mode="nontrainable"
            )
        elif self.model_name == 'UNETR':
            model = UNETR(
                in_channels=self.in_channels,
                out_channels=self.out_channels,
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
                in_channels=self.in_channels,
                out_channels=self.out_channels,
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
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                channels=(32, 64, 128, 256),
                strides=(2, 2, 2),
                dropout=0.1
            )
        elif self.model_name == 'VNet':
            model = VNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
                act=("elu", {"inplace": True}),
                dropout_prob_down=0.5,
                dropout_prob_up=(0.5, 0.5),
                dropout_dim=3,
                bias=False
            )
        elif self.model_name == 'HighResNet':
            model = HighResNet(
                spatial_dims=3,
                in_channels=self.in_channels,
                out_channels=self.out_channels,
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
        elif self.model_name == 'VNet3D':
            model = self._create_vnet3d()
        else:
            raise ValueError(f"不支持的模型类型: {self.model_name}")
            
        return model.to(self.device)
    
    def _create_vnet3d(self):
        """
        创建自定义VNet3D网络架构
        基于V-Net设计，针对3D医学图像分割优化
        """
        class VNet3DBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
                super().__init__()
                self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding)
                self.bn1 = nn.BatchNorm3d(out_channels)
                self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size, 1, padding)
                self.bn2 = nn.BatchNorm3d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                
                # 残差连接
                if in_channels != out_channels or stride != 1:
                    self.shortcut = nn.Sequential(
                        nn.Conv3d(in_channels, out_channels, 1, stride),
                        nn.BatchNorm3d(out_channels)
                    )
                else:
                    self.shortcut = nn.Identity()
                    
            def forward(self, x):
                residual = self.shortcut(x)
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += residual
                return self.relu(out)
        
        class VNet3D(nn.Module):
            def __init__(self, in_channels, out_channels):
                super().__init__()
                
                # 编码器
                self.encoder1 = VNet3DBlock(in_channels, 32)
                self.encoder2 = VNet3DBlock(32, 64, stride=2)
                self.encoder3 = VNet3DBlock(64, 128, stride=2)
                self.encoder4 = VNet3DBlock(128, 256, stride=2)
                self.encoder5 = VNet3DBlock(256, 512, stride=2)
                
                # 瓶颈层
                self.bottleneck = VNet3DBlock(512, 1024, stride=2)
                
                # 解码器
                self.upconv5 = nn.ConvTranspose3d(1024, 512, 2, 2)
                self.decoder5 = VNet3DBlock(1024, 512)  # 512 + 512 from skip connection
                
                self.upconv4 = nn.ConvTranspose3d(512, 256, 2, 2)
                self.decoder4 = VNet3DBlock(512, 256)  # 256 + 256 from skip connection
                
                self.upconv3 = nn.ConvTranspose3d(256, 128, 2, 2)
                self.decoder3 = VNet3DBlock(256, 128)  # 128 + 128 from skip connection
                
                self.upconv2 = nn.ConvTranspose3d(128, 64, 2, 2)
                self.decoder2 = VNet3DBlock(128, 64)  # 64 + 64 from skip connection
                
                self.upconv1 = nn.ConvTranspose3d(64, 32, 2, 2)
                self.decoder1 = VNet3DBlock(64, 32)  # 32 + 32 from skip connection
                
                # 输出层
                self.final_conv = nn.Conv3d(32, out_channels, 1)
                
                # Dropout层
                self.dropout = nn.Dropout3d(0.1)
                
            def forward(self, x):
                # 编码器路径
                enc1 = self.encoder1(x)
                enc2 = self.encoder2(enc1)
                enc3 = self.encoder3(enc2)
                enc4 = self.encoder4(enc3)
                enc5 = self.encoder5(enc4)
                
                # 瓶颈层
                bottleneck = self.bottleneck(enc5)
                bottleneck = self.dropout(bottleneck)
                
                # 解码器路径
                dec5 = self.upconv5(bottleneck)
                dec5 = torch.cat([dec5, enc5], dim=1)
                dec5 = self.decoder5(dec5)
                dec5 = self.dropout(dec5)
                
                dec4 = self.upconv4(dec5)
                dec4 = torch.cat([dec4, enc4], dim=1)
                dec4 = self.decoder4(dec4)
                
                dec3 = self.upconv3(dec4)
                dec3 = torch.cat([dec3, enc3], dim=1)
                dec3 = self.decoder3(dec3)
                
                dec2 = self.upconv2(dec3)
                dec2 = torch.cat([dec2, enc2], dim=1)
                dec2 = self.decoder2(dec2)
                
                dec1 = self.upconv1(dec2)
                dec1 = torch.cat([dec1, enc1], dim=1)
                dec1 = self.decoder1(dec1)
                
                # 输出
                output = self.final_conv(dec1)
                
                return output
        
        return VNet3D(self.in_channels, self.out_channels)
    
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
            ),
            'froc': self._create_froc_metric()
        }
        
        return metrics
    
    def _create_froc_metric(self):
        """动态创建FROC指标以避免循环导入"""
        try:
            from evaluate import FROCMetric
            return FROCMetric(
                distance_threshold=5.0,
                include_background=False
            )
        except ImportError:
            print("警告: 无法导入FROCMetric，跳过FROC指标")
            return None
        
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
    return ['UNet', 'SegResNet', 'UNETR', 'SwinUNETR', 'AttentionUNet', 'VNet', 'HighResNet', 'VNet3D']


class SpecializedModelFactory:
    """
    知识蒸馏、融合网络、神经架构搜索和NAS-蒸馏集成
    支持多数据集动态通道配置
    """
    
    def __init__(self, 
                 model_type: str = 'fusion',  # 'fusion', 'distillation', 'nas', 'nas_distillation'
                 device: str = 'auto',
                 dataset_type: str = 'BraTS',
                 **kwargs):
        self.model_type = model_type
        self.device = self._setup_device(device)
        self.dataset_type = dataset_type
        self.kwargs = kwargs
        # 将数据集类型传递给kwargs，供子模型使用
        self.kwargs['dataset_type'] = dataset_type
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
        """创建复合架构模型"""
        if self.model_type == 'fusion':
            return self._create_fusion_model()
        elif self.model_type == 'distillation':
            return self._create_distillation_model()
        elif self.model_type == 'nas':
            return self._create_nas_model()
        elif self.model_type == 'nas_distillation':
            return self._create_nas_distillation_model()
        else:
            raise ValueError(f"不支持的复合架构模型类型: {self.model_type}")
            
    def _create_fusion_model(self):
        """创建融合网络模型"""
        # 获取基础模型列表，默认使用所有7个网络架构
        base_model_names = self.kwargs.get('base_models', get_all_supported_models())
        base_models = []
        
        for model_name in base_model_names:
            basic_model = BasicModelBank(model_name=model_name, device=self.device, dataset_type=self.dataset_type)
            base_models.append(basic_model.model)
            
        # 根据数据集类型获取输出类别数
        dataset_configs = {
            'BraTS': 4,
            'MS_MultiSpine': 6
        }
        default_num_classes = dataset_configs.get(self.dataset_type, 4)
        
        # 创建融合网络
        fusion_model = FusionNetworkArchitecture(
            base_models=base_models,
            fusion_channels=self.kwargs.get('fusion_channels', [64, 128, 256, 512]),
            num_classes=self.kwargs.get('num_classes', default_num_classes),
            fusion_type=self.kwargs.get('fusion_type', 'cross_attention')
        )
        
        return fusion_model.to(self.device)
        
    def _create_distillation_model(self):
        """创建知识蒸馏模型"""
        distillation_type = self.kwargs.get('distillation_type', 'multi_teacher')
        
        if distillation_type == 'multi_teacher':
            # 获取学生模型名称
            student_name = self.kwargs.get('student_model', 'VNet3D')
            
            # 创建教师模型，默认使用所有网络架构作为教师模型，但排除学生模型
            all_models = get_all_supported_models()
            teacher_names = self.kwargs.get('teacher_models', all_models)
            
            # 确保教师模型列表中不包含学生模型，避免重复训练
            if isinstance(teacher_names, list):
                teacher_names = [name for name in teacher_names if name != student_name]
            
            # 如果过滤后教师模型列表为空，使用除学生模型外的所有模型
            if not teacher_names:
                teacher_names = [name for name in all_models if name != student_name]
                print(f"⚠ 警告：教师模型列表为空，自动使用除学生模型({student_name})外的所有模型作为教师模型")
            
            print(f"📚 知识蒸馏配置：")
            print(f"  学生模型: {student_name}")
            print(f"  教师模型: {teacher_names}")
            print(f"  教师模型数量: {len(teacher_names)}")
            
            teacher_models = []
            
            for teacher_name in teacher_names:
                teacher = BasicModelBank(model_name=teacher_name, device=self.device, dataset_type=self.dataset_type)
                
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
                
            # 创建学生模型（使用之前获取的student_name）
            student = BasicModelBank(model_name=student_name, device=self.device, dataset_type=self.dataset_type)
            
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
            student_name = self.kwargs.get('student_model', 'VNet3D')
            
            # 确保教师模型和学生模型不相同
            if teacher_name == student_name:
                # 如果相同，自动选择不同的教师模型
                all_models = get_all_supported_models()
                available_teachers = [name for name in all_models if name != student_name]
                if available_teachers:
                    teacher_name = available_teachers[0]  # 选择第一个可用的教师模型
                    print(f"⚠ 警告：教师模型与学生模型相同，自动选择 {teacher_name} 作为教师模型")
                else:
                    raise ValueError(f"无法找到与学生模型 {student_name} 不同的教师模型")
            
            print(f"📚 渐进式知识蒸馏配置：")
            print(f"  教师模型: {teacher_name}")
            print(f"  学生模型: {student_name}")
            
            teacher = BasicModelBank(model_name=teacher_name, device=self.device, dataset_type=self.dataset_type)
            student = BasicModelBank(model_name=student_name, device=self.device, dataset_type=self.dataset_type)
            
            distillation_model = ProgressiveKnowledgeDistillation(
                teacher_models=[teacher.model],  # 渐进式蒸馏使用单个教师模型的列表
                student_model=student.model,
                device=self.device
            )
            
        else:
            raise ValueError(f"不支持的蒸馏类型: {distillation_type}")
            
        # 对于nn.Module类型的模型，移动到指定设备
        if hasattr(distillation_model, 'to'):
            return distillation_model.to(self.device)
        else:
            # 对于非nn.Module类型（如ProgressiveKnowledgeDistillation），直接返回
            return distillation_model
        
    def _create_nas_model(self):
        """创建神经架构搜索模型"""
        nas_type = self.kwargs.get('nas_type', 'supernet')
        
        if nas_type == 'supernet':
            # 根据数据集类型获取默认配置
            dataset_configs = {
                'BraTS': {'in_channels': 4, 'num_classes': 4},
                'MS_MultiSpine': {'in_channels': 2, 'num_classes': 6}
            }
            default_config = dataset_configs.get(self.dataset_type, {'in_channels': 4, 'num_classes': 4})
            
            # 创建超网络
            supernet = SuperNet(
                in_channels=self.kwargs.get('in_channels', default_config['in_channels']),
                num_classes=self.kwargs.get('num_classes', default_config['num_classes']),
                base_channels=self.kwargs.get('base_channels', 32),
                num_layers=self.kwargs.get('num_layers', 4)
            )
            return supernet.to(self.device)
            
        elif nas_type == 'searcher':
            # 根据数据集类型获取默认配置
            dataset_configs = {
                'BraTS': {'in_channels': 4, 'num_classes': 4},
                'MS_MultiSpine': {'in_channels': 2, 'num_classes': 6}
            }
            default_config = dataset_configs.get(self.dataset_type, {'in_channels': 4, 'num_classes': 4})
            
            # 创建DARTS搜索器
            supernet = SuperNet(
                in_channels=self.kwargs.get('in_channels', default_config['in_channels']),
                num_classes=self.kwargs.get('num_classes', default_config['num_classes']),
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
            
    def _create_nas_distillation_model(self):
        """创建NAS-蒸馏集成模型"""
        from nas_distillation import NASDistillationIntegration
        
        # 获取教师模型列表，默认使用所有基础模型
        teacher_models = self.kwargs.get('teacher_models', get_all_supported_models())
        
        # 确保教师模型列表不为空
        if not teacher_models:
            teacher_models = get_all_supported_models()
            print(f"⚠ 警告：教师模型列表为空，自动使用所有基础模型作为教师模型")
            
        print(f"🔬 NAS-蒸馏集成配置：")
        print(f"  教师模型: {teacher_models}")
        print(f"  教师模型数量: {len(teacher_models)}")
        print(f"  数据集类型: {self.dataset_type}")
        
        # 创建NAS-蒸馏集成器
        nas_distillation = NASDistillationIntegration(
            teacher_models=teacher_models,
            device=self.device,
            dataset_type=self.dataset_type,
            nas_epochs=self.kwargs.get('nas_epochs', 50),
            distillation_epochs=self.kwargs.get('distillation_epochs', 100),
            arch_lr=self.kwargs.get('arch_lr', 3e-4),
            model_lr=self.kwargs.get('model_lr', 1e-3),
            distillation_lr=self.kwargs.get('distillation_lr', 1e-4),
            temperature=self.kwargs.get('temperature', 4.0),
            alpha=self.kwargs.get('alpha', 0.7),
            save_dir=self.kwargs.get('save_dir', './checkpoints/nas_distillation')
        )
        
        return nas_distillation
            
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
    支持多数据集动态通道配置
    """
    
    @staticmethod
    def create_model(model_config: Dict[str, Any]) -> Any:
        """
        根据配置创建模型
        """
        model_category = model_config.get('category', 'basic')  # 'basic', 'advanced'
        dataset_type = model_config.get('dataset_type', 'BraTS')
        
        if model_category == 'basic':
            return BasicModelBank(
                model_name=model_config.get('model_name', 'UNet'),
                device=model_config.get('device', 'auto'),
                dataset_type=dataset_type
            )
            
        elif model_category == 'advanced':
            return SpecializedModelFactory(
                model_type=model_config.get('model_type', 'fusion'),
                device=model_config.get('device', 'auto'),
                dataset_type=dataset_type,
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
    
    # 显示复合架构模型信息
    model_info = ModelFactory.get_model_info()
    print(f"\n复合架构模型信息:")
    for category, models in model_info.items():
        print(f"  {category}: {models}")
    
    # 创建测试模型验证设备配置
    test_model = BasicModelBank()
    device = test_model._setup_device('auto')
    print(f"\n设备配置验证通过，设备: {device}")
    
    # 测试模型工厂
    try:
        # 测试基础模型创建（BraTS数据集）
        basic_config_brats = {
            'category': 'basic',
            'model_name': 'UNet',
            'device': 'auto',
            'dataset_type': 'BraTS'
        }
        basic_model_brats = ModelFactory.create_model(basic_config_brats)
        print(f"BraTS基础模型创建成功: {type(basic_model_brats).__name__}")
        
        # 测试基础模型创建（MS_MultiSpine数据集）
        basic_config_ms = {
            'category': 'basic',
            'model_name': 'UNet',
            'device': 'auto',
            'dataset_type': 'MS_MultiSpine'
        }
        basic_model_ms = ModelFactory.create_model(basic_config_ms)
        print(f"MS_MultiSpine基础模型创建成功: {type(basic_model_ms).__name__}")
        
        # 测试复合架构模型创建（融合网络）
        fusion_config = {
            'category': 'advanced',
            'model_type': 'fusion',
            'device': 'auto',
            'dataset_type': 'BraTS',
            'kwargs': {
                'base_models': ['UNet', 'SegResNet'],
                'fusion_type': 'cross_attention'
            }
        }
        fusion_model = ModelFactory.create_model(fusion_config)
        print(f"融合模型创建成功: {type(fusion_model).__name__}")
        
        print("\n所有功能验证通过！")
        
    except Exception as e:
        print(f"\n功能验证失败: {e}")
        print("注意: 某些功能可能需要额外的依赖或数据")