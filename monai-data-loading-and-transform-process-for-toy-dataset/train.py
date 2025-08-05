import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path
from DatasetLoader_transforms import DatasetLoader
from model import (
    BasicModelBank, create_optimizer, create_scheduler,
    SpecializedModelFactory, ModelFactory
)
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose

# 导入中文字体配置
try:
    from font_config import configure_chinese_font
    # 自动配置中文字体
    configure_chinese_font()
except ImportError:
    import warnings
    warnings.warn("未找到font_config模块，中文显示可能出现问题", UserWarning)

class ModelTrainer:
    """
    医学图像分割模型训练器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 设置随机种子
        set_determinism(seed=config.get('seed', 42))
        
        # 统一创建输出目录结构
        self._setup_output_directories()
        
        # 初始化组件
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_logging()
        

        
    def _setup_data(self):
        """
        设置数据加载器
        """
        print("设置数据加载器...")
        
        data_loader = DatasetLoader(
            data_dir=self.config['data_dir'],
            cache_rate=self.config.get('cache_rate', 0.1),
            num_workers=self.config.get('num_workers', 4),
            seed=self.config.get('seed', 42)
        )
        
        self.train_loader, self.val_loader = data_loader.get_dataloaders(
            batch_size=self.config.get('batch_size', 2)
        )
        
        print(f"训练批次数: {len(self.train_loader)}")
        print(f"验证批次数: {len(self.val_loader)}")
        
    def _setup_model(self):
        """
        设置模型、损失函数和指标
        统一使用自适应损失函数策略和完整评估指标
        支持单模型训练和高级模型训练两种模式
        """
        print("设置模型...")
        print("[策略] 所有模型类型统一使用:")
        print("  - 自适应组合损失函数策略 (DiceCE + Focal + Tversky + GeneralizedDice + DiceFocal)")
        print("  - 完整评估指标集合 (Dice + Hausdorff + SurfaceDistance + ConfusionMatrix + MeanIoU + GeneralizedDiceScore)")
        
        # 检查是否使用高级模型
        model_category = self.config.get('model_category', 'basic')  # 'basic', 'advanced'
        
        if model_category == 'advanced':
            print(f"\n[模型类型] 高级模型: {self.config.get('model_type', 'fusion').upper()}")
            
            # 使用ModelFactory创建高级模型
            self.advanced_model = ModelFactory.create_model({
                'category': 'advanced',
                'model_type': self.config.get('model_type', 'fusion'),
                'device': self.device,
                'kwargs': self.config.get('model_kwargs', {})
            })
            
            self.model = self.advanced_model.get_model()
            self.loss_function = self.advanced_model.get_loss_function()
            
            # 为高级模型创建基础指标
            temp_model_creator = BasicModelBank(device=self.device)
            self.metrics = temp_model_creator.get_metrics()
            
            self.is_advanced = True
            
            print(f"[高级模型] 类型: {self.config.get('model_type', 'fusion')}")
            print(f"[参数统计] 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
            
        else:
            model_name = self.config.get('model_name', 'unet')
            print(f"\n[模型类型] 单个模型: {model_name.upper()}")
            
            self.model_creator = BasicModelBank(
                model_name=model_name,
                device=self.device
            )
            
            self.model = self.model_creator.get_model()
            self.loss_function = self.model_creator.get_loss_function()
            self.metrics = self.model_creator.get_metrics()
            self.is_advanced = False
            
            print(f"[参数统计] 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 验证损失函数和指标配置
        if hasattr(self, 'is_advanced') and self.is_advanced:
            # 高级模型训练模式
            loss_info = {
                'strategy': 'Advanced Model Loss',
                'type': self.config.get('model_type', 'fusion'),
                'description': f'Advanced {self.config.get("model_type", "fusion")} model with specialized loss function'
            }
        else:
            # 单模型训练模式
            loss_info = self.model_creator.get_loss_info()
        print(f"\n[损失函数] 策略: {loss_info['strategy']}")
        print(f"[损失函数] 类型: {loss_info['type']}")
        print(f"[损失函数] 描述: {loss_info['description']}")
        
        print(f"[评估指标] 启用指标数量: {len(self.metrics)}")
        print(f"[评估指标] 指标列表: {list(self.metrics.keys())}")
        
        # 后处理变换
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([AsDiscrete(to_onehot=4)])
        
        print(f"\n[设备配置] 计算设备: {self.device}")
        print("[配置完成] 模型、损失函数和评估指标设置完成\n")
        
    def _setup_training(self):
        """
        设置训练组件
        """
        print("设置训练组件...")
        
        # 单模型或高级模型优化器
        if hasattr(self, 'is_advanced') and self.is_advanced:
            # 高级模型可能需要特殊的优化器设置
            model_type = self.config.get('model_type', 'fusion')
            
            if model_type == 'nas':
                # NAS模型需要分别优化架构参数和模型参数
                if hasattr(self.model, 'get_arch_parameters'):
                    self.arch_optimizer = torch.optim.Adam(
                        self.model.get_arch_parameters(),
                        lr=self.config.get('arch_lr', 3e-4),
                        betas=(0.5, 0.999),
                        weight_decay=1e-3
                    )
                    
                    self.optimizer = torch.optim.SGD(
                        self.model.get_model_parameters(),
                        lr=self.config.get('learning_rate', 1e-3),
                        momentum=0.9,
                        weight_decay=3e-4
                    )
                else:
                    # 如果是NAS搜索器，使用其内置的优化器
                    self.optimizer = None  # NAS搜索器自己管理优化器
                    
            elif model_type == 'distillation':
                # 知识蒸馏可能需要不同的学习率
                self.optimizer = create_optimizer(
                    self.model,
                    optimizer_name=self.config.get('optimizer', 'adamw'),
                    learning_rate=self.config.get('learning_rate', 5e-5),  # 通常较小的学习率
                    weight_decay=self.config.get('weight_decay', 1e-5)
                )
            else:
                # 融合网络使用标准优化器
                self.optimizer = create_optimizer(
                    self.model,
                    optimizer_name=self.config.get('optimizer', 'adamw'),
                    learning_rate=self.config.get('learning_rate', 1e-4),
                    weight_decay=self.config.get('weight_decay', 1e-5)
                )
                    
            # 学习率调度器
            if self.optimizer is not None:
                self.scheduler = create_scheduler(
                    self.optimizer,
                    scheduler_name=self.config.get('scheduler', 'cosineannealinglr'),
                    T_max=self.config.get('max_epochs', 500)
                )
                
            if hasattr(self, 'arch_optimizer'):
                self.arch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.arch_optimizer,
                    T_max=self.config.get('max_epochs', 500)
                )
                
        else:
                # 标准单模型优化器
                self.optimizer = create_optimizer(
                    self.model,
                    optimizer_name=self.config.get('optimizer', 'adamw'),
                    learning_rate=self.config.get('learning_rate', 1e-4),
                    weight_decay=self.config.get('weight_decay', 1e-5)
                )
                
                # 学习率调度器
                self.scheduler = create_scheduler(
                    self.optimizer,
                    scheduler_name=self.config.get('scheduler', 'cosineannealinglr'),
                    T_max=self.config.get('max_epochs', 500)
                )
        
        # 早停和模型检查点
        self.early_stopping = EarlyStopping(
            patience=self.config.get('patience', 20),
            min_delta=self.config.get('min_delta', 0.001)
        )
        
        self.checkpoint = ModelCheckpoint(
            save_dir=self.checkpoints_dir,
            filename='best_model.pth'
        )
        
        # 指标跟踪器
        self.metrics_tracker = MetricsTracker()
        
    def _setup_output_directories(self):
        """
        统一设置输出目录结构
        所有模型（基础和高级）都使用相同的目录结构
        """
        base_output_dir = Path(self.config.get('output_dir', './outputs'))
        
        # 确定模型类型和名称
        model_category = self.config.get('model_category', 'basic')
        
        if model_category == 'advanced':
            # 高级模型：./outputs/models/{model_type}_model/
            model_type = self.config.get('model_type', 'fusion')
            model_dir_name = f"{model_type}_model"
        else:
            # 基础模型：./outputs/models/{model_name}/
            model_name = self.config.get('model_name', 'unet')
            model_dir_name = model_name
        
        # 统一的模型输出目录结构
        self.output_dir = base_output_dir / 'models' / model_dir_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建子目录
        self.checkpoints_dir = self.output_dir / 'checkpoints'
        self.logs_dir = self.output_dir / 'logs'
        self.metrics_dir = self.output_dir / 'metrics'
        self.visualizations_dir = self.output_dir / 'visualizations'
        
        # 创建所有子目录
        for dir_path in [self.checkpoints_dir, self.logs_dir, self.metrics_dir, self.visualizations_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"模型输出目录: {self.output_dir}")
        print(f"  - 检查点: {self.checkpoints_dir}")
        print(f"  - 日志: {self.logs_dir}")
        print(f"  - 指标: {self.metrics_dir}")
        print(f"  - 可视化: {self.visualizations_dir}")
    
    def _setup_logging(self):
        """
        设置日志记录
        """
        # TensorBoard日志保存到logs/tensorboard_logs目录
        tensorboard_dir = self.logs_dir / 'tensorboard_logs'
        tensorboard_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=tensorboard_dir)
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            训练指标字典
        """
        self.model.train()
        
        epoch_loss = 0
        num_batches = len(self.train_loader)
        
        # 重置指标
        for metric in self.metrics.values():
            metric.reset()
        
        progress_bar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}")
        
        for batch_idx, batch_data in enumerate(progress_bar):
            inputs = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            
            # 单模型或高级模型训练
            if hasattr(self, 'is_advanced') and self.is_advanced:
                # 高级模型训练
                model_type = self.config.get('model_type', 'fusion')
                
                if model_type == 'nas' and hasattr(self.advanced_model, 'train_step'):
                    # NAS模型使用特殊的训练步骤
                    try:
                        loss_value = self.advanced_model.train_step(
                            inputs, labels, self.optimizer
                        )
                        epoch_loss += loss_value
                    except Exception as e:
                        print(f"NAS训练步骤失败: {e}")
                        # 回退到标准训练
                        if self.optimizer is not None:
                            self.optimizer.zero_grad()
                            outputs = self.model(inputs)
                            loss = self.loss_function(outputs, labels)
                            loss.backward()
                            self.optimizer.step()
                            epoch_loss += loss.item()
                                
                elif model_type == 'distillation':
                    # 知识蒸馏训练
                    if hasattr(self.advanced_model, 'train_step'):
                        loss_value = self.advanced_model.train_step(
                            inputs, labels, self.optimizer
                        )
                        epoch_loss += loss_value
                    else:
                        # 标准蒸馏训练
                        self.optimizer.zero_grad()
                        
                        if hasattr(self.model, 'forward') and callable(getattr(self.model, 'forward')):
                            # 检查模型是否是知识蒸馏模型（需要labels参数）
                            if hasattr(self.model, '__class__') and 'MultiTeacherDistillation' in str(self.model.__class__):
                                # 知识蒸馏模型需要inputs和labels两个参数
                                outputs, loss = self.model(inputs, labels)
                            else:
                                # 检查模型是否返回学生和教师输出
                                model_output = self.model(inputs)
                                if isinstance(model_output, tuple) and len(model_output) == 2:
                                    student_output, teacher_outputs = model_output
                                    loss = self.loss_function(student_output, teacher_outputs, labels)
                                else:
                                    # 标准输出
                                    outputs = model_output
                                    loss = self.loss_function(outputs, labels)
                        else:
                            outputs = self.model(inputs)
                            loss = self.loss_function(outputs, labels)
                            
                        loss.backward()
                        self.optimizer.step()
                        epoch_loss += loss.item()
                        
                else:
                    # 融合网络或其他高级模型
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)
                    loss.backward()
                    self.optimizer.step()
                    epoch_loss += loss.item()
                        
                # 对于高级模型，尝试获取输出用于指标计算
                try:
                    if 'outputs' not in locals():
                        # 检查是否是知识蒸馏模型
                        if hasattr(self.model, '__class__') and 'MultiTeacherDistillation' in str(self.model.__class__):
                            outputs, _ = self.model(inputs, labels)  # 知识蒸馏模型返回(outputs, loss)
                        else:
                            outputs = self.model(inputs)
                        
                    # 计算指标
                    outputs_list = decollate_batch(outputs)
                    labels_list = decollate_batch(labels)
                    
                    outputs_convert = [self.post_pred(pred) for pred in outputs_list]
                    labels_convert = [self.post_label(label) for label in labels_list]
                    
                    # 更新指标
                    self.metrics['dice'](y_pred=outputs_convert, y=labels_convert)
                except Exception as e:
                    print(f"高级模型指标计算失败: {e}")
                    
                # 更新进度条
                current_loss = epoch_loss / (batch_idx + 1)
                if self.optimizer is not None:
                    lr_info = f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                else:
                    lr_info = 'NAS-managed'
                    
                progress_bar.set_postfix({
                    'Loss': f'{current_loss:.4f}',
                    'LR': lr_info,
                    'Type': model_type.upper()
                })
                
            else:
                    # 标准单模型训练
                    self.optimizer.zero_grad()
                    outputs = self.model(inputs)
                    
                    # 计算损失
                    loss = self.loss_function(outputs, labels)
                    
                    # 反向传播
                    loss.backward()
                    self.optimizer.step()
                    
                    # 累计损失
                    epoch_loss += loss.item()
                    
                    # 计算指标
                    outputs_list = decollate_batch(outputs)
                    labels_list = decollate_batch(labels)
                    
                    outputs_convert = [self.post_pred(pred) for pred in outputs_list]
                    labels_convert = [self.post_label(label) for label in labels_list]
                    
                    # 更新指标
                    self.metrics['dice'](y_pred=outputs_convert, y=labels_convert)
                    
                    # 更新进度条
                    current_loss = epoch_loss / (batch_idx + 1)
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'LR': f'{self.optimizer.param_groups[0]["lr"]:.2e}'
                    })
        
        # 计算epoch平均指标
        avg_loss = epoch_loss / num_batches
        dice_scores = self.metrics['dice'].aggregate()
        avg_dice = dice_scores.mean().item()  # 计算所有类别的平均Dice分数
        
        return {
            'loss': avg_loss,
            'dice': avg_dice,
            'learning_rate': self.optimizer.param_groups[0]['lr']
        }
    
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        """
        验证一个epoch
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            验证指标字典
        """
        self.model.eval()
        
        epoch_loss = 0
        num_batches = len(self.val_loader)
        
        # 重置指标
        for metric in self.metrics.values():
            metric.reset()
        
        progress_bar = tqdm(self.val_loader, desc=f"验证 Epoch {epoch+1}")
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(progress_bar):
                inputs = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                
                # 使用滑动窗口推理
                if hasattr(self, 'model_creator'):
                    outputs = self.model_creator.sliding_window_inference(inputs)
                elif hasattr(self, 'is_advanced') and self.is_advanced:
                    # 高级模型直接使用模型进行推理
                    outputs = self.model(inputs)
                else:
                    outputs = self.model(inputs)
                
                # 计算损失
                loss = self.loss_function(outputs, labels)
                epoch_loss += loss.item()
                
                # 计算指标
                outputs_list = decollate_batch(outputs)
                labels_list = decollate_batch(labels)
                
                outputs_convert = [self.post_pred(pred) for pred in outputs_list]
                labels_convert = [self.post_label(label) for label in labels_list]
                
                # 更新指标
                self.metrics['dice'](y_pred=outputs_convert, y=labels_convert)
                
                # 更新进度条
                current_loss = epoch_loss / (batch_idx + 1)
                progress_bar.set_postfix({'Val Loss': f'{current_loss:.4f}'})
        
        # 计算epoch平均指标
        avg_loss = epoch_loss / num_batches
        dice_scores = self.metrics['dice'].aggregate()
        avg_dice = dice_scores.mean().item()  # 计算所有类别的平均Dice分数
        
        return {
            'val_loss': avg_loss,
            'val_dice': avg_dice
        }
    
    def train(self):
        """
        执行完整的训练流程
        """
        print("开始训练...")
        print(f"最大训练轮数: {self.config.get('max_epochs', 500)}")
        print(f"输出目录: {self.output_dir}")
        print("-" * 50)
        
        max_epochs = self.config.get('max_epochs', 500)
        best_metric = -1
        
        for epoch in range(max_epochs):
            start_time = time.time()
            
            # 更新自适应损失函数的epoch信息
            if hasattr(self, 'model_creator') and hasattr(self.model_creator, 'update_loss_epoch'):
                self.model_creator.update_loss_epoch(epoch, max_epochs)
            elif hasattr(self, 'is_advanced') and self.is_advanced and hasattr(self.advanced_model, 'update_loss_epoch'):
                self.advanced_model.update_loss_epoch(epoch, max_epochs)
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()
            
            # 记录指标
            all_metrics = {**train_metrics, **val_metrics}
            self.metrics_tracker.update(all_metrics)
            
            # 记录到TensorBoard
            for key, value in all_metrics.items():
                self.writer.add_scalar(f'Metrics/{key}', value, epoch)
            
            # 保存最佳模型
            current_metric = val_metrics['val_dice']
            if current_metric > best_metric:
                best_metric = current_metric
                
                # 保存单模型
                single_state = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_metric': best_metric,
                    'config': self.config,
            
                    'is_parallel': False
                }
                self.checkpoint.save(single_state)
                    
                print(f"保存最佳模型 (Dice: {best_metric:.4f})")
            
            # 早停检查
            if self.early_stopping(val_metrics['val_loss']):
                print(f"早停触发，在第 {epoch+1} 轮停止训练")
                break
            
            # 打印epoch结果
            epoch_time = time.time() - start_time
            print(f"Epoch {epoch+1}/{max_epochs} - {epoch_time:.1f}s")
            print(f"  训练 - Loss: {train_metrics['loss']:.4f}, Dice: {train_metrics['dice']:.4f}")
            print(f"  验证 - Loss: {val_metrics['val_loss']:.4f}, Dice: {val_metrics['val_dice']:.4f}")
            print(f"  学习率: {train_metrics['learning_rate']:.2e}")
            print("-" * 50)
        
        # 保存训练历史到metrics目录
        self.metrics_tracker.save_history(self.metrics_dir / 'training_history.json')
        
        # 绘制训练曲线到visualizations目录
        self.plot_training_curves()
        
        # 生成样本预测可视化
        self.generate_sample_predictions()
        
        print(f"训练完成！最佳Dice系数: {best_metric:.4f}")
        print(f"模型和日志保存在: {self.output_dir}")
        print(f"  - 模型权重: {self.checkpoints_dir}")
        print(f"  - 训练历史: {self.metrics_dir / 'training_history.json'}")
        print(f"  - 训练曲线: {self.visualizations_dir / 'training_curves.png'}")
        print(f"  - 样本预测: {self.visualizations_dir / 'sample_predictions.png'}")
        print(f"  - TensorBoard日志: {self.logs_dir / 'tensorboard_logs'}")
        
        self.writer.close()
    
    def plot_training_curves(self):
        """
        绘制训练曲线
        """
        history = self.metrics_tracker.get_history()
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程监控', fontsize=16)
        
        # 损失曲线
        axes[0, 0].plot(history['loss'], label='训练损失', color='blue')
        axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
        axes[0, 0].set_title('损失曲线')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Dice系数曲线
        axes[0, 1].plot(history['dice'], label='训练Dice', color='blue')
        axes[0, 1].plot(history['val_dice'], label='验证Dice', color='red')
        axes[0, 1].set_title('Dice系数曲线')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Dice Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # 学习率曲线
        axes[1, 0].plot(history['learning_rate'], label='学习率', color='green')
        axes[1, 0].set_title('学习率变化')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # 验证指标对比
        axes[1, 1].plot(history['val_loss'], label='验证损失', color='red')
        ax2 = axes[1, 1].twinx()
        ax2.plot(history['val_dice'], label='验证Dice', color='orange')
        axes[1, 1].set_title('验证指标对比')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Loss', color='red')
        ax2.set_ylabel('Validation Dice', color='orange')
        axes[1, 1].legend(loc='upper left')
        ax2.legend(loc='upper right')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        curves_path = self.visualizations_dir / 'training_curves.png'
        plt.savefig(curves_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: {curves_path}")
    
    def generate_sample_predictions(self):
        """
        生成样本预测可视化
        """
        try:
            print("生成样本预测可视化...")
            
            # 设置模型为评估模式
            self.model.eval()
            
            # 从验证集获取一个批次的数据
            with torch.no_grad():
                for batch_data in self.val_loader:
                    inputs = batch_data['image'].to(self.device)
                    labels = batch_data['label'].to(self.device)
                    
                    # 进行推理
                    if hasattr(self, 'model_creator'):
                        outputs = self.model_creator.sliding_window_inference(inputs)
                    else:
                        outputs = self.model(inputs)
                    
                    # 只处理第一个样本
                    input_sample = inputs[0].cpu().numpy()
                    label_sample = labels[0].cpu().numpy()
                    output_sample = torch.softmax(outputs[0], dim=0).cpu().numpy()
                    pred_sample = np.argmax(output_sample, axis=0)
                    
                    # 创建可视化
                    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                    fig.suptitle('样本预测可视化', fontsize=16)
                    
                    # 选择中间切片进行显示
                    slice_idx = input_sample.shape[-1] // 2
                    
                    # 显示输入图像的不同模态
                    modalities = ['FLAIR', 'T1w', 'T1gd', 'T2w']
                    for i in range(min(4, input_sample.shape[0])):
                        row, col = i // 2, i % 2
                        if row < 2 and col < 2:
                            axes[row, col].imshow(input_sample[i, :, :, slice_idx], cmap='gray')
                            axes[row, col].set_title(f'输入 - {modalities[i] if i < len(modalities) else f"模态{i+1}"}')
                            axes[row, col].axis('off')
                    
                    # 显示真实标签
                    if len(label_sample.shape) == 4:  # one-hot编码
                        label_display = np.argmax(label_sample, axis=0)
                    else:
                        label_display = label_sample
                    
                    axes[0, 2].imshow(label_display[:, :, slice_idx], cmap='tab10', vmin=0, vmax=3)
                    axes[0, 2].set_title('真实标签')
                    axes[0, 2].axis('off')
                    
                    # 显示预测结果
                    axes[1, 2].imshow(pred_sample[:, :, slice_idx], cmap='tab10', vmin=0, vmax=3)
                    axes[1, 2].set_title('预测结果')
                    axes[1, 2].axis('off')
                    
                    # 保存可视化
                    predictions_path = self.visualizations_dir / 'sample_predictions.png'
                    plt.tight_layout()
                    plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    
                    print(f"样本预测可视化已保存: {predictions_path}")
                    break  # 只处理第一个批次
                    
        except Exception as e:
            print(f"生成样本预测可视化时出错: {e}")
            # 创建一个简单的占位图
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            ax.text(0.5, 0.5, f'样本预测可视化生成失败\n错误: {str(e)}', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.axis('off')
            
            predictions_path = self.visualizations_dir / 'sample_predictions.png'
            plt.savefig(predictions_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"占位图已保存: {predictions_path}")

def main():
    pass

if __name__ == "__main__":
    main()

# 配置如下：
# config = {
#     'data_dir': "BraTS2024-BraTS-GLI-TrainingData/training_data1_v2",
#     'batch_size': 2,
#     'cache_rate': 0.1,
#     'num_workers': 2,
#     'model_name': 'unet',

#     'max_epochs': 500,
#     'learning_rate': 1e-4,
#     'weight_decay': 1e-5,
#     'optimizer': 'adamw',
#     'scheduler': 'cosine',
#     'patience': 20,
#     'min_delta': 0.001,
#     'device': 'cuda' if torch.cuda.is_available() else 'cpu',
#     'seed': 42,
#     'output_dir': './outputs'
# }