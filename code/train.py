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
from model import BasicModelBank, BankModelIntegration, create_optimizer, create_scheduler, create_full_ensemble
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose

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
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_logging()
        
    def create_ensemble_model(self):
        """
        创建集成模型，将多个训练好的模型融合成一个集成模型
        """
        if 'trained_models' not in self.config:
            print("[错误] 没有找到训练好的模型信息")
            return
            
        trained_models = self.config['trained_models']
        print(f"\n[集成] 开始创建集成模型，包含{len(trained_models)}个训练好的模型")
        
        # 加载训练好的模型
        models = []
        for model_info in trained_models:
            model_name = model_info['name']
            model_path = model_info['path']
            
            if os.path.exists(model_path):
                print(f"[集成] 加载模型: {model_name} from {model_path}")
                
                # 创建模型实例
                model = BasicModelBank(
                    model_name=model_name,
                    device=self.device,
                    use_adaptive_loss=True,
                    use_full_metrics=True
                )
                
                # 加载训练好的权重
                checkpoint = torch.load(model_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    model.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.model.load_state_dict(checkpoint)
                    
                model.model.eval()
                models.append(model)
            else:
                print(f"[警告] 模型文件不存在: {model_path}")
        
        if len(models) > 1:
            # 创建集成模型
            ensemble_model = BankModelIntegration(models, device=self.device)
            
            # 保存集成模型
            ensemble_path = os.path.join(self.config['output_dir'], 'ensemble_model.pth')
            os.makedirs(os.path.dirname(ensemble_path), exist_ok=True)
            
            # 保存集成模型信息
            ensemble_info = {
                'model_type': 'ensemble',
                'num_models': len(models),
                'model_names': [info['name'] for info in trained_models],
                'ensemble_method': 'average_voting',
                'created_time': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            
            torch.save({
                'ensemble_info': ensemble_info,
                'model_configs': [info['config'] for info in trained_models]
            }, ensemble_path)
            
            print(f"[成功] 集成模型已创建并保存到: {ensemble_path}")
            print(f"[集成] 融合方法: 平均预测 + 投票机制")
            print(f"[集成] 包含模型: {[info['name'] for info in trained_models]}")
        else:
            print(f"[错误] 需要至少2个模型才能创建集成，当前只有{len(models)}个模型")
        
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
        支持单模型训练、顺序集成训练和并行集成训练三种模式
        """
        print("设置模型...")
        print("[策略] 所有模型类型统一使用:")
        print("  - 自适应组合损失函数策略 (DiceCE + Focal + Tversky + GeneralizedDice + DiceFocal)")
        print("  - 完整评估指标集合 (Dice + Hausdorff + SurfaceDistance + ConfusionMatrix + MeanIoU + GeneralizedDiceScore)")
        
        # 检查是否使用集成模型
        use_ensemble = self.config.get('use_ensemble', False)
        model_names = self.config.get('model_names', [])
        ensemble_mode = self.config.get('ensemble_mode', 'parallel')
        
        if use_ensemble and model_names and ensemble_mode == 'parallel':
            print(f"\n[模型类型] 并行集成训练 - {len(model_names)}个模型")
            print(f"[模型列表] {', '.join([name.upper() for name in model_names])}")
            
            # 创建多个模型实例
            self.models = {}
            self.model_creators = {}
            total_params = 0
            
            for model_name in model_names:
                model_creator = BasicModelBank(
                    model_name=model_name,
                    device=self.device
                )
                self.model_creators[model_name] = model_creator
                self.models[model_name] = model_creator.get_model()
                
                # 计算每个模型的参数数量
                model_params = sum(p.numel() for p in self.models[model_name].parameters())
                total_params += model_params
                print(f"[{model_name.upper()}] 参数数量: {model_params:,}")
            
            # 使用第一个模型的损失函数和指标作为统一配置
            first_model_creator = list(self.model_creators.values())[0]
            self.loss_function = first_model_creator.get_loss_function()
            self.metrics = first_model_creator.get_metrics()
            self.is_ensemble = True
            self.ensemble_mode = 'parallel'
            
            print(f"[参数统计] 总参数数量: {total_params:,}")
            print(f"[训练模式] 并行训练{len(model_names)}个模型")
            
        elif use_ensemble and (ensemble_mode == 'sequential' or not model_names):
            print("\n[模型类型] 逐个集成训练")
            self.ensemble_model = create_full_ensemble(device=self.device)
            self.is_ensemble = True
            self.ensemble_mode = 'sequential'
            
            # 为集成模型使用与单模型相同的高级指标和损失函数
            # 创建临时模型实例来获取标准化的损失函数和指标
            temp_model_creator = BasicModelBank(device=self.device)
            self.loss_function = temp_model_creator.get_loss_function()
            self.metrics = temp_model_creator.get_metrics()
            
            print(f"[集成配置] 包含 {len(self.ensemble_model.models)} 个子模型")
            print(f"[集成配置] 每个子模型都使用自适应损失函数")
            
            # 计算总参数数量
            total_params = sum(sum(p.numel() for p in model.model.parameters()) 
                             for model in self.ensemble_model.models)
            print(f"[参数统计] 总参数数量: {total_params:,}")
            
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
            self.is_ensemble = False
            
            print(f"[参数统计] 模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 验证损失函数和指标配置
        if use_ensemble and model_names:
            # 并行集成训练模式，使用第一个模型的信息
            loss_info = first_model_creator.get_loss_info()
        elif use_ensemble:
            # 顺序集成训练模式
            loss_info = temp_model_creator.get_loss_info()
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
        
        if self.is_ensemble:
            model_names = self.config.get('model_names', [])
            
            if model_names and hasattr(self, 'models'):
                # 并行多模型训练
                self.optimizers = {}
                self.schedulers = {}
                
                for model_name in model_names:
                    optimizer = create_optimizer(
                        self.models[model_name],
                        optimizer_name=self.config.get('optimizer', 'adamw'),
                        learning_rate=self.config.get('learning_rate', 1e-4),
                        weight_decay=self.config.get('weight_decay', 1e-5)
                    )
                    self.optimizers[model_name] = optimizer
                    
                    scheduler = create_scheduler(
                        optimizer,
                        scheduler_name=self.config.get('scheduler', 'cosineannealinglr'),
                        T_max=self.config.get('max_epochs', 500)
                    )
                    self.schedulers[model_name] = scheduler
                    
                print(f"为 {len(self.optimizers)} 个模型创建了优化器")
            else:
                # 传统集成模型训练
                self.optimizers = []
                self.schedulers = []
                
                for model in self.ensemble_model.models:
                    optimizer = create_optimizer(
                        model.model,
                        optimizer_name=self.config.get('optimizer', 'adamw'),
                        learning_rate=self.config.get('learning_rate', 1e-4),
                        weight_decay=self.config.get('weight_decay', 1e-5)
                    )
                    self.optimizers.append(optimizer)
                    
                    scheduler = create_scheduler(
                        optimizer,
                        scheduler_name=self.config.get('scheduler', 'cosineannealinglr'),
                        T_max=self.config.get('max_epochs', 500)
                    )
                    self.schedulers.append(scheduler)
                    
                print(f"为 {len(self.optimizers)} 个模型创建了优化器")
        else:
            # 单模型优化器
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
            save_dir=self.output_dir / 'checkpoints',
            filename='best_model.pth'
        )
        
        # 指标跟踪器
        self.metrics_tracker = MetricsTracker()
        
    def _setup_logging(self):
        """
        设置日志记录
        """
        self.writer = SummaryWriter(log_dir=self.output_dir / 'tensorboard')
        
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            epoch: 当前epoch数
            
        Returns:
            训练指标字典
        """
        if self.is_ensemble:
            if hasattr(self, 'ensemble_mode') and self.ensemble_mode == 'parallel':
                # 并行多模型训练
                for model in self.models.values():
                    model.train()
            else:
                # 逐个集成模型训练
                for model in self.ensemble_model.models:
                    model.model.train()
        else:
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
            
            if self.is_ensemble:
                if hasattr(self, 'ensemble_mode') and self.ensemble_mode == 'parallel':
                    # 并行多模型训练
                    model_names = self.config.get('model_names', [])
                    total_loss = 0
                    ensemble_outputs = []
                    
                    # 为每个模型计算损失并更新
                    for model_name in model_names:
                        model = self.models[model_name]
                        optimizer = self.optimizers[model_name]
                        
                        optimizer.zero_grad()
                        
                        # 前向传播
                        outputs = model(inputs)
                        ensemble_outputs.append(outputs)
                        
                        # 计算损失
                        loss = self.loss_function(outputs, labels)
                        total_loss += loss.item()
                        
                        # 反向传播
                        loss.backward()
                        optimizer.step()
                    
                    # 使用集成预测计算指标
                    ensemble_pred = torch.mean(torch.stack(ensemble_outputs), dim=0)
                    epoch_loss += total_loss / len(model_names)
                    
                    # 计算指标
                    outputs_list = decollate_batch(ensemble_pred)
                    labels_list = decollate_batch(labels)
                    
                    outputs_convert = [self.post_pred(pred) for pred in outputs_list]
                    labels_convert = [self.post_label(label) for label in labels_list]
                    
                    # 更新指标
                    self.metrics['dice'](y_pred=outputs_convert, y=labels_convert)
                    
                    # 更新进度条
                    current_loss = epoch_loss / (batch_idx + 1)
                    avg_lr = sum(self.optimizers[name].param_groups[0]['lr'] for name in model_names) / len(model_names)
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg_LR': f'{avg_lr:.2e}'
                    })
                    
                else:
                    # 逐个集成模型训练
                    total_loss = 0
                    ensemble_outputs = []
                    
                    # 为每个子模型计算损失并更新
                    for i, (model, optimizer) in enumerate(zip(self.ensemble_model.models, self.optimizers)):
                        optimizer.zero_grad()
                        
                        # 前向传播
                        outputs = model.model(inputs)
                        ensemble_outputs.append(outputs)
                        
                        # 计算损失
                        loss = self.loss_function(outputs, labels)
                        total_loss += loss.item()
                        
                        # 反向传播
                        loss.backward()
                        optimizer.step()
                    
                    # 使用集成预测计算指标
                    ensemble_pred = torch.mean(torch.stack(ensemble_outputs), dim=0)
                    epoch_loss += total_loss / len(self.ensemble_model.models)
                    
                    # 计算指标
                    outputs_list = decollate_batch(ensemble_pred)
                    labels_list = decollate_batch(labels)
                    
                    outputs_convert = [self.post_pred(pred) for pred in outputs_list]
                    labels_convert = [self.post_label(label) for label in labels_list]
                    
                    # 更新指标
                    self.metrics['dice'](y_pred=outputs_convert, y=labels_convert)
                    
                    # 更新进度条
                    current_loss = epoch_loss / (batch_idx + 1)
                    avg_lr = sum(opt.param_groups[0]['lr'] for opt in self.optimizers) / len(self.optimizers)
                    progress_bar.set_postfix({
                        'Loss': f'{current_loss:.4f}',
                        'Avg_LR': f'{avg_lr:.2e}'
                    })
                
            else:
                # 单模型训练
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
        
        if self.is_ensemble:
            avg_lr = sum(opt.param_groups[0]['lr'] for opt in self.optimizers) / len(self.optimizers)
            return {
                'loss': avg_loss,
                'dice': avg_dice,
                'learning_rate': avg_lr
            }
        else:
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
        if self.is_ensemble:
            if hasattr(self, 'ensemble_mode') and self.ensemble_mode == 'parallel':
                # 并行多模型验证
                for model in self.models.values():
                    model.eval()
            else:
                # 逐个集成模型验证
                for model in self.ensemble_model.models:
                    model.model.eval()
        else:
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
                
                if self.is_ensemble:
                    if hasattr(self, 'ensemble_mode') and self.ensemble_mode == 'parallel':
                        # 并行多模型预测
                        model_names = self.config.get('model_names', [])
                        ensemble_outputs = []
                        for model_name in model_names:
                            model = self.models[model_name]
                            model_creator = self.model_creators[model_name]
                            model_outputs = model_creator.sliding_window_inference(inputs)
                            ensemble_outputs.append(model_outputs)
                        
                        # 集成预测结果
                        outputs = torch.mean(torch.stack(ensemble_outputs), dim=0)
                    else:
                        # 使用逐个集成模型预测
                        outputs = self.ensemble_model.predict(inputs)
                else:
                    # 使用滑动窗口推理
                    outputs = self.model_creator.sliding_window_inference(inputs)
                
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
            if self.is_ensemble:
                if hasattr(self, 'ensemble_mode') and self.ensemble_mode == 'parallel':
                    # 并行多模型：为每个模型更新损失函数epoch信息
                    model_names = self.config.get('model_names', [])
                    for model_name in model_names:
                        model_creator = self.model_creators[model_name]
                        if hasattr(model_creator, 'update_loss_epoch'):
                            model_creator.update_loss_epoch(epoch, max_epochs)
                else:
                    # 逐个集成模型：为每个子模型更新损失函数epoch信息
                    for model in self.ensemble_model.models:
                        if hasattr(model, 'update_loss_epoch'):
                            model.update_loss_epoch(epoch, max_epochs)
            else:
                # 单模型：更新损失函数epoch信息
                if hasattr(self.model_creator, 'update_loss_epoch'):
                    self.model_creator.update_loss_epoch(epoch, max_epochs)
            
            # 训练
            train_metrics = self.train_epoch(epoch)
            
            # 验证
            val_metrics = self.validate_epoch(epoch)
            
            # 更新学习率
            if self.is_ensemble:
                if hasattr(self, 'ensemble_mode') and self.ensemble_mode == 'parallel':
                    # 并行多模型：更新每个模型的学习率调度器
                    model_names = self.config.get('model_names', [])
                    for model_name in model_names:
                        scheduler = self.schedulers[model_name]
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_metrics['val_loss'])
                        else:
                            scheduler.step()
                else:
                    # 逐个集成模型：更新每个子模型的学习率调度器
                    for scheduler in self.schedulers:
                        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            scheduler.step(val_metrics['val_loss'])
                        else:
                            scheduler.step()
            else:
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
                
                if self.is_ensemble:
                    if hasattr(self, 'ensemble_mode') and self.ensemble_mode == 'parallel':
                        # 保存并行多模型
                        model_names = self.config.get('model_names', [])
                        parallel_state = {
                            'epoch': epoch,
                            'model_states': {name: self.models[name].state_dict() for name in model_names},
                            'optimizer_states': {name: self.optimizers[name].state_dict() for name in model_names},
                            'scheduler_states': {name: self.schedulers[name].state_dict() for name in model_names},
                            'best_metric': best_metric,
                            'config': self.config,
                            'is_ensemble': True,
                            'is_parallel': True,
                            'ensemble_mode': 'parallel',
                            'model_names': model_names
                        }
                        self.checkpoint.save(parallel_state)
                    else:
                        # 保存逐个集成模型
                        ensemble_state = {
                            'epoch': epoch,
                            'model_states': [model.model.state_dict() for model in self.ensemble_model.models],
                            'optimizer_states': [opt.state_dict() for opt in self.optimizers],
                            'scheduler_states': [sch.state_dict() for sch in self.schedulers],
                            'best_metric': best_metric,
                            'config': self.config,
                            'is_ensemble': True,
                            'is_parallel': False,
                            'ensemble_mode': 'sequential',
                            'model_names': [model.model_name for model in self.ensemble_model.models]
                        }
                        self.checkpoint.save(ensemble_state)
                else:
                    # 保存单模型
                    single_state = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scheduler_state_dict': self.scheduler.state_dict(),
                        'best_metric': best_metric,
                        'config': self.config,
                        'is_ensemble': False,
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
        
        # 保存训练历史
        self.metrics_tracker.save_history(self.output_dir / 'training_history.json')
        
        # 绘制训练曲线
        self.plot_training_curves()
        
        print(f"训练完成！最佳Dice系数: {best_metric:.4f}")
        print(f"模型和日志保存在: {self.output_dir}")
        
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
        plt.savefig(self.output_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"训练曲线已保存: {self.output_dir / 'training_curves.png'}")

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
#     'use_ensemble': True,
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