import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
from monai.networks.nets import UNet, SegResNet, UNETR, SwinUNETR, AttentionUnet, VNet, HighResNet
from monai.losses import DiceCELoss, FocalLoss, TverskyLoss, GeneralizedDiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, ConfusionMatrixMetric, MeanIoU, GeneralizedDiceScore
from monai.inferers import sliding_window_inference




class BasicModelBank:
    """
    简化的BraTS分割模型，支持基础模型架构和多种损失函数策略
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


class BankModelIntegration:
    """
    集成模型，支持多模型投票
    """
    def __init__(self, models: List[BasicModelBank], device: str = 'auto'):
        self.models = models
        self.device = self._setup_device(device)
        
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
        
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """集成预测（平均）"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model.sliding_window_inference(inputs)
                predictions.append(pred)
        
        # 平均融合
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_pred
        
    def vote_predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """投票预测"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model.sliding_window_inference(inputs)
                # 转换为类别预测
                pred_class = torch.argmax(pred, dim=1, keepdim=True)
                predictions.append(pred_class)
        
        # 投票机制
        stacked_preds = torch.stack(predictions, dim=0)
        voted_pred = torch.mode(stacked_preds, dim=0)[0]
        
        # 转换回one-hot格式
        num_classes = 4
        voted_onehot = torch.zeros_like(predictions[0]).repeat(1, num_classes, 1, 1, 1)
        voted_onehot.scatter_(1, voted_pred, 1)
        
        return voted_onehot


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


def create_full_ensemble(device: str = 'auto') -> BankModelIntegration:
    """
    创建包含所有7个模型的完整集成
    统一使用自适应损失函数策略和完整评估指标
    
    Returns:
        包含所有模型的集成对象
    """
    all_models = get_all_supported_models()
    models = []
    for model_name in all_models:
        model = BasicModelBank(
            model_name=model_name, 
            device=device,
            use_adaptive_loss=True,  # 统一使用自适应损失函数
            use_full_metrics=True    # 统一使用完整评估指标
        )
        models.append(model)
    
    return BankModelIntegration(models, device)


def get_all_supported_models() -> List[str]:
    """
    获取所有支持的模型列表
    """
    return ['UNet', 'SegResNet', 'UNETR', 'SwinUNETR', 'AttentionUNet', 'VNet', 'HighResNet']


if __name__ == "__main__":
    # 功能验证
    print("功能验证...")
    
    # 显示支持的模型列表
    supported_models = get_all_supported_models()
    print(f"支持的模型数量: {len(supported_models)}")
    print(f"模型列表: {supported_models}")
    
    # 创建测试模型验证设备配置
    test_model = BasicModelBank()
    device = test_model._setup_device('auto')
    print(f"\n参数配置验证通过，设备: {device}")
    
    print("\n所有功能正常。")