import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import logging
import os
import json
import time
from pathlib import Path
from torch.cuda.amp import GradScaler
from torch.amp import autocast
import gc
from tqdm import tqdm
from colorama import init, Fore, Back, Style
import psutil

from nas_search import SuperNet, DARTSSearcher
from knowledge_distillation import KnowledgeDistillationLoss, MultiTeacherDistillation
from model import ModelFactory

# 初始化colorama用于彩色终端输出
init(autoreset=True)

class NASDistillationIntegration:
    """
    NAS-蒸馏集成类：实现两阶段训练流程
    阶段1：NAS搜索找到最优架构
    阶段2：使用最优架构作为学生模型进行知识蒸馏训练
    """
    
    def __init__(self,
                 teacher_models: List[str],
                 device: torch.device,
                 dataset_type: str = 'BraTS',
                 nas_epochs: int = 50,
                 distillation_epochs: int = 100,
                 arch_lr: float = 3e-4,
                 model_lr: float = 1e-3,
                 distillation_lr: float = 1e-4,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 save_dir: str = './checkpoints/nas_distillation',
                 # NAS相关参数
                 nas_type: str = 'darts',
                 base_channels: int = 8,
                 num_layers: int = 3,
                 max_layers: int = 8,
                 start_layers: int = 1,
                 # 知识蒸馏参数映射
                 distillation_temperature: Optional[float] = None,
                 distillation_alpha: Optional[float] = None,
                 # 教师模型预训练参数
                 teacher_pretrain_epochs: int = 50,
                 teacher_learning_rate: float = 1e-4,
                 # 其他参数
                 auto_adjust: bool = False):
        """
        初始化NAS-蒸馏集成器
        
        Args:
            teacher_models: 教师模型列表
            device: 计算设备
            dataset_type: 数据集类型
            nas_epochs: NAS搜索轮数
            distillation_epochs: 知识蒸馏轮数
            arch_lr: 架构参数学习率
            model_lr: 模型参数学习率
            distillation_lr: 蒸馏阶段学习率
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            save_dir: 模型保存目录
        """
        self.teacher_models = teacher_models
        self.device = device
        self.dataset_type = dataset_type
        self.nas_epochs = nas_epochs
        self.distillation_epochs = distillation_epochs
        self.arch_lr = arch_lr
        self.model_lr = model_lr
        self.distillation_lr = distillation_lr
        
        # 知识蒸馏参数映射处理
        self.temperature = distillation_temperature if distillation_temperature is not None else temperature
        self.alpha = distillation_alpha if distillation_alpha is not None else alpha
        
        # NAS相关参数
        self.nas_type = nas_type
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.max_layers = max_layers
        self.start_layers = start_layers
        self.auto_adjust = auto_adjust
        
        # 教师模型预训练参数
        self.teacher_pretrain_epochs = teacher_pretrain_epochs
        self.teacher_learning_rate = teacher_learning_rate
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置数据集相关参数
        if dataset_type == 'MS_MultiSpine':
            self.in_channels = 2
            self.num_classes = 6
        else:  # BraTS
            self.in_channels = 4
            self.num_classes = 4
            
        # 初始化组件
        self.supernet = None
        self.searcher = None
        self.teacher_ensemble = None
        self.student_model = None
        self.distillation_model = None
        
        # 训练状态
        self.nas_completed = False
        self.best_architecture = None
        
        # 内存优化设置
        self._setup_memory_optimization()
        
        # 混合精度训练设置
        self.use_amp = torch.cuda.is_available()
        self.scaler = GradScaler() if self.use_amp else None
        
        # 内存监控设置
        self.memory_monitor_interval = 10  # 每10个batch监控一次内存
        self.auto_cleanup_threshold = 8.0  # 内存使用超过8GB时自动清理
        
        logging.info(f"初始化NAS-蒸馏集成器: 教师模型={teacher_models}, 数据集={dataset_type}")
        logging.info(f"混合精度训练: {'启用' if self.use_amp else '禁用'}")
        logging.info(f"内存监控: 间隔={self.memory_monitor_interval}batch, 自动清理阈值={self.auto_cleanup_threshold}GB")
        
        # 训练进度跟踪
        self.training_start_time = None
        self.current_phase = "初始化"
        self.phase_start_time = None
        
    def _setup_memory_optimization(self):
        """
        设置内存优化配置
        """
        # 启用梯度检查点以减少内存使用
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # 设置CUDA内存分配策略
        if torch.cuda.is_available():
            # 启用可扩展段以减少内存碎片
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,max_split_size_mb:512'
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            
            # 设置内存分配策略以减少碎片
            torch.cuda.set_per_process_memory_fraction(0.85)  # 限制GPU内存使用为85%
            
            # 启用内存历史记录用于调试
            try:
                torch.cuda.memory._record_memory_history(enabled=True, alloc_trace_record_context=True)
            except:
                pass
            
            logging.info("[内存优化] 已启用CUDA内存优化设置")
            logging.info(f"[内存优化] GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            logging.info("[内存优化] 已设置GPU内存使用限制为85%")
            logging.info("[内存优化] 已启用内存历史记录和自动清理机制")
        
    def _clear_memory_cache(self):
        """
        强化的GPU内存缓存清理
        """
        # 多次强制垃圾回收
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 尝试内存碎片整理
            try:
                torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
                torch.cuda.memory._record_memory_history(enabled=None)
            except:
                pass
            
            # 记录详细内存使用情况
            allocated = torch.cuda.memory_allocated() / 1024**3  # GB
            reserved = torch.cuda.memory_reserved() / 1024**3    # GB
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
            max_reserved = torch.cuda.max_memory_reserved() / 1024**3    # GB
            
            logging.info(f"GPU内存状态: 当前分配 {allocated:.2f}GB, 当前保留 {reserved:.2f}GB")
            logging.info(f"GPU内存峰值: 最大分配 {max_allocated:.2f}GB, 最大保留 {max_reserved:.2f}GB")
            
            # 如果内存使用过高，发出警告
            if allocated > 10.0:  # 超过10GB
                logging.warning(f"GPU内存使用过高: {allocated:.2f}GB，可能存在内存泄漏")
            
            # 重置内存统计
            torch.cuda.reset_peak_memory_stats()
        
    def initialize_nas_search(self) -> None:
        """
        初始化NAS搜索组件
        """
        logging.info("初始化NAS搜索组件...")
        
        # 创建超网络（使用传入的参数）
        self.supernet = SuperNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=self.base_channels,
            num_layers=self.num_layers,
            dataset_type=self.dataset_type
        )
        
        # 启用梯度检查点以减少内存使用
        if hasattr(self.supernet, 'enable_checkpointing'):
            self.supernet.enable_checkpointing()
        
        # 为所有模块启用梯度检查点
        for module in self.supernet.modules():
            if hasattr(module, 'gradient_checkpointing'):
                module.gradient_checkpointing = True
        
        # 清理内存缓存
        self._clear_memory_cache()
        
        # 创建DARTS搜索器
        self.searcher = DARTSSearcher(
            supernet=self.supernet,
            device=self.device,
            arch_lr=self.arch_lr,
            model_lr=self.model_lr
        )
        
        logging.info("NAS搜索组件初始化完成")
        
    def initialize_teacher_models(self) -> None:
        """
        初始化教师模型集合
        """
        logging.info("初始化教师模型集合...")
        
        # 创建模型工厂
        model_factory = ModelFactory()
        
        # 加载教师模型
        teachers = {}
        for model_name in self.teacher_models:
            try:
                model_config = {
                    'category': 'basic',
                    'model_name': model_name,
                    'device': self.device,
                    'dataset_type': self.dataset_type
                }
                model_bank = model_factory.create_model(model_config)
                model = model_bank.model
                model = model.to(self.device)
                model.eval()  # 教师模型设为评估模式
                teachers[model_name] = model
                logging.info(f"成功加载教师模型: {model_name}")
            except Exception as e:
                logging.warning(f"加载教师模型 {model_name} 失败: {e}")
                
        if not teachers:
            raise ValueError("没有成功加载任何教师模型")
            
        self.teacher_ensemble = teachers
        logging.info(f"教师模型集合初始化完成，共加载 {len(teachers)} 个模型")
        
    def pretrain_teacher_models(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        预训练教师模型（如果需要）
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            预训练结果
        """
        # 打印阶段标题
        self._print_phase_header(
            "教师模型预训练阶段", 
            f"教师模型: {', '.join(self.teacher_models)} | 检查预训练权重"
        )
        
        pretrain_results = {
            'pretrained_models': [],
            'skipped_models': [],
            'individual_results': {},
            'total_time': 0
        }
        
        pretrain_start_time = time.time()
        
        # 初始化教师模型（如果还没有初始化）
        if not hasattr(self, 'teacher_ensemble') or not self.teacher_ensemble:
            self.initialize_teacher_models()
        
        # 检查每个教师模型是否需要预训练
        models_to_pretrain = []
        for model_name in self.teacher_models:
            pretrained_path = self.save_dir.parent / 'pretrained_teachers' / model_name / 'models' / 'best_model.pth'
            if not pretrained_path.exists():
                models_to_pretrain.append(model_name)
                print(f"{Fore.YELLOW}⚠️  教师模型 {model_name} 需要预训练{Style.RESET_ALL}")
            else:
                pretrain_results['skipped_models'].append(model_name)
                print(f"{Fore.GREEN}✅ 教师模型 {model_name} 已有预训练权重{Style.RESET_ALL}")
        
        if not models_to_pretrain:
            print(f"{Fore.GREEN}✅ 所有教师模型都已有预训练权重，跳过预训练阶段!{Style.RESET_ALL}")
        else:
            print(f"{Fore.CYAN}🚀 开始预训练 {len(models_to_pretrain)} 个教师模型...{Style.RESET_ALL}")
            
            # 为每个需要预训练的模型执行预训练
            for idx, model_name in enumerate(models_to_pretrain, 1):
                model_start_time = time.time()
                
                print(f"\n{Fore.MAGENTA}📍 教师模型 {idx}/{len(models_to_pretrain)}: {model_name}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
                
                # 执行单个教师模型的预训练
                model_result = self._pretrain_single_teacher(
                    model_name, train_loader, val_loader, idx, len(models_to_pretrain)
                )
                
                pretrain_results['individual_results'][model_name] = model_result
                pretrain_results['pretrained_models'].append(model_name)
                
                model_time = time.time() - model_start_time
                print(f"{Fore.GREEN}✅ 教师模型 {model_name} 预训练完成! 耗时: {self._format_time(model_time)}{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
        
        # 最终统计
        total_time = time.time() - pretrain_start_time
        pretrain_results['total_time'] = total_time
        
        print(f"{Fore.GREEN}🎉 教师模型预训练阶段完成!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 预训练统计:{Style.RESET_ALL}")
        print(f"   • 教师模型总数: {len(self.teacher_models)}")
        print(f"   • 预训练模型: {len(pretrain_results['pretrained_models'])}")
        print(f"   • 跳过模型: {len(pretrain_results['skipped_models'])}")
        print(f"   • 总耗时: {self._format_time(total_time)}")
        print(f"   • 内存使用: {self._get_memory_info_str()}")
        
        if pretrain_results['individual_results']:
            print(f"   • 各模型详情:")
            for model_name, result in pretrain_results['individual_results'].items():
                print(f"     - {model_name}: 最佳损失 {result['best_val_loss']:.4f}, 耗时 {self._format_time(result['training_time'])}")
        
        return pretrain_results
    
    def _pretrain_single_teacher(self, model_name: str, train_loader, val_loader, 
                                model_idx: int, total_models: int) -> Dict[str, Any]:
        """
        预训练单个教师模型
        
        Args:
            model_name: 教师模型名称
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            model_idx: 当前模型索引
            total_models: 总模型数量
            
        Returns:
            单个模型的预训练结果
        """
        from torch.optim import Adam
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        import torch.nn.functional as F
        
        # 获取教师模型
        teacher_model = self.teacher_ensemble[model_name]
        teacher_model.train()
        
        # 设置预训练参数
        pretrain_epochs = getattr(self, 'teacher_pretrain_epochs', 50)  # 默认50轮
        learning_rate = getattr(self, 'teacher_learning_rate', 1e-4)  # 默认学习率
        
        print(f"{Fore.WHITE}配置: 轮数={pretrain_epochs}, 学习率={learning_rate}, 设备={self.device}{Style.RESET_ALL}")
        
        # 创建优化器和调度器
        optimizer = Adam(teacher_model.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=False)
        
        # 训练统计
        best_val_loss = float('inf')
        training_stats = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'best_epoch': 0,
            'training_time': 0
        }
        
        model_start_time = time.time()
        
        # 创建主进度条
        epoch_pbar = tqdm(
            range(pretrain_epochs),
            desc=f"{Fore.BLUE}🎓 {model_name} 预训练{Style.RESET_ALL}",
            ncols=120,
            leave=True,
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # 训练阶段
            teacher_model.train()
            train_loss = 0.0
            train_batches = 0
            
            try:
                for batch_idx, batch in enumerate(train_loader):
                    data, target = batch['image'].to(self.device), batch['label'].to(self.device)
                    
                    # 处理目标张量维度
                    if target.dim() == 5 and target.size(1) == 1:
                        target = target.squeeze(1)
                    target = target.long()
                    
                    # 验证target张量值范围
                    target_min, target_max = target.min().item(), target.max().item()
                    if target_min < 0 or target_max >= self.num_classes:
                        target = torch.clamp(target, 0, self.num_classes - 1)
                    
                    # 前向传播和反向传播
                    optimizer.zero_grad()
                    
                    if self.use_amp:
                        with autocast('cuda'):
                            output = teacher_model(data)
                            loss = F.cross_entropy(output, target)
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        output = teacher_model(data)
                        loss = F.cross_entropy(output, target)
                        loss.backward()
                        optimizer.step()
                    
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # 内存监控
                    if batch_idx % self.memory_monitor_interval == 0:
                        self._monitor_and_cleanup_memory()
                        
            except RuntimeError as e:
                if "DataLoader worker" in str(e):
                    logging.error(f"教师模型 {model_name} 训练DataLoader错误: {e}")
                    train_loader = self._recreate_dataloader(train_loader, num_workers=0)
                    continue
                else:
                    raise e
            
            # 验证阶段
            teacher_model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                try:
                    for batch in val_loader:
                        data, target = batch['image'].to(self.device), batch['label'].to(self.device)
                        
                        # 处理目标张量维度
                        if target.dim() == 5 and target.size(1) == 1:
                            target = target.squeeze(1)
                        target = target.long()
                        
                        # 验证target张量值范围
                        target_min, target_max = target.min().item(), target.max().item()
                        if target_min < 0 or target_max >= self.num_classes:
                            target = torch.clamp(target, 0, self.num_classes - 1)
                        
                        if self.use_amp:
                            with autocast('cuda'):
                                output = teacher_model(data)
                                batch_loss = F.cross_entropy(output, target).item()
                        else:
                            output = teacher_model(data)
                            batch_loss = F.cross_entropy(output, target).item()
                        
                        val_loss += batch_loss
                        val_batches += 1
                        
                except RuntimeError as e:
                    if "DataLoader worker" in str(e):
                        logging.error(f"教师模型 {model_name} 验证DataLoader错误: {e}")
                        val_loader = self._recreate_dataloader(val_loader, num_workers=0)
                        continue
                    else:
                        raise e
            
            # 计算平均损失
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            epoch_time = time.time() - epoch_start_time
            
            # 记录统计信息
            training_stats['train_losses'].append(train_loss)
            training_stats['val_losses'].append(val_loss)
            training_stats['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # 更新主进度条
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss:.4f}',
                'Val_Loss': f'{val_loss:.4f}',
                'Best': f'{best_val_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Time': f'{epoch_time:.1f}s',
                'Mem': self._get_memory_info_str().split('|')[0].strip()
            })
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_stats['best_epoch'] = epoch
                
                # 显示新的最佳结果
                tqdm.write(f"{Fore.GREEN}✨ {model_name} 新的最佳验证损失: {val_loss:.4f} (Epoch {epoch+1}){Style.RESET_ALL}")
                
                # 保存最优教师模型
                teacher_save_dir = self.save_dir.parent / 'pretrained_teachers' / model_name / 'models'
                teacher_save_dir.mkdir(parents=True, exist_ok=True)
                teacher_save_path = teacher_save_dir / 'best_model.pth'
                
                torch.save({
                    'model_state_dict': teacher_model.state_dict(),
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'model_name': model_name,
                    'dataset_type': self.dataset_type,
                    'training_config': {
                        'learning_rate': learning_rate,
                        'epochs': pretrain_epochs,
                        'optimizer': 'Adam'
                    }
                }, teacher_save_path)
            
            # 学习率调度
            scheduler.step(val_loss)
            
            # 每个epoch结束后清理内存缓存
            self._clear_memory_cache()
        
        # 关闭进度条
        epoch_pbar.close()
        
        # 记录训练时间
        training_stats['training_time'] = time.time() - model_start_time
        training_stats['best_val_loss'] = best_val_loss
        
        # 保存训练统计
        stats_save_dir = self.save_dir.parent / 'pretrained_teachers' / model_name
        stats_save_dir.mkdir(parents=True, exist_ok=True)
        stats_save_path = stats_save_dir / 'training_stats.json'
        
        with open(stats_save_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats
    
    def _format_time(self, seconds: float) -> str:
        """
        格式化时间显示
        
        Args:
            seconds: 秒数
            
        Returns:
            格式化的时间字符串
        """
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def search_architecture(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        执行NAS架构搜索
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            最优架构信息
        """
        if not self.supernet or not self.searcher:
            self.initialize_nas_search()
        
        # 打印阶段标题
        self._print_phase_header(
            "NAS架构搜索阶段", 
            f"搜索轮数: {self.nas_epochs} | 架构学习率: {self.arch_lr} | 模型学习率: {self.model_lr}"
        )
        
        best_val_loss = float('inf')
        best_arch_params = None
        search_start_time = time.time()
        
        # 创建主进度条
        epoch_pbar = tqdm(
            range(self.nas_epochs),
            desc=f"{Fore.GREEN}🔍 NAS搜索{Style.RESET_ALL}",
            ncols=120,
            leave=True,
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            
            # 训练阶段
            self.supernet.train()
            train_loss = 0.0
            train_batches = 0
            
            # 创建训练批次进度条
            train_pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"{Fore.BLUE}  📚 训练{Style.RESET_ALL}",
                leave=False,
                position=1,
                ncols=100,
                disable=len(train_loader) < 10  # 如果batch数量太少，禁用子进度条
            )
            
            try:
                for batch_idx, batch in train_pbar:
                    data, target = batch['image'].to(self.device), batch['label'].to(self.device)
                    
                    # 处理目标张量维度：移除多余的通道维度
                    if target.dim() == 5 and target.size(1) == 1:
                        target = target.squeeze(1)  # 从[B, 1, H, W, D]变为[B, H, W, D]
                    
                    # 确保目标张量为Long类型（交叉熵损失要求）
                    target = target.long()
                    
                    # 验证target张量值范围，防止CUDA设备端断言错误
                    target_min, target_max = target.min().item(), target.max().item()
                    if target_min < 0 or target_max >= self.num_classes:
                        logging.warning(f"Target张量值超出范围: min={target_min}, max={target_max}, num_classes={self.num_classes}")
                        # 将target值限制在有效范围内
                        target = torch.clamp(target, 0, self.num_classes - 1)
                    
                    # 检查并处理NaN或无穷值
                    if torch.isnan(target).any() or torch.isinf(target).any():
                        logging.error("Target张量包含NaN或无穷值，跳过此batch")
                        continue
                    
                    # 更新模型参数（使用混合精度训练）
                    self.searcher.model_optimizer.zero_grad()
                    
                    if self.use_amp:
                        with autocast('cuda'):
                            output = self.supernet(data)
                            loss = F.cross_entropy(output, target)
                        
                        self.scaler.scale(loss).backward()
                        self.scaler.step(self.searcher.model_optimizer)
                        self.scaler.update()
                    else:
                        output = self.supernet(data)
                        loss = F.cross_entropy(output, target)
                        loss.backward()
                        self.searcher.model_optimizer.step()
                    # 更新架构参数（每隔几个batch，使用混合精度训练）
                    if batch_idx % 5 == 0:
                        self.searcher.arch_optimizer.zero_grad()
                        
                        if self.use_amp:
                            with autocast('cuda'):
                                arch_output = self.supernet(data)
                                arch_loss = F.cross_entropy(arch_output, target)
                            
                            self.scaler.scale(arch_loss).backward()
                            self.scaler.step(self.searcher.arch_optimizer)
                            self.scaler.update()
                        else:
                            arch_output = self.supernet(data)
                            arch_loss = F.cross_entropy(arch_output, target)
                            arch_loss.backward()
                            self.searcher.arch_optimizer.step()
                    train_loss += loss.item()
                    train_batches += 1
                    
                    # 更新训练进度条
                    train_pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Avg': f'{train_loss/train_batches:.4f}',
                        'Mem': self._get_memory_info_str().split('|')[0].strip()
                    })
                    
                    # 内存监控和自动清理
                    if batch_idx % self.memory_monitor_interval == 0:
                        self._monitor_and_cleanup_memory()
                        
            except RuntimeError as e:
                if "DataLoader worker" in str(e):
                    logging.error(f"DataLoader worker错误: {e}")
                    logging.info("尝试重新创建DataLoader...")
                    # 重新创建train_loader，强制设置num_workers=0
                    train_loader = self._recreate_dataloader(train_loader, num_workers=0)
                    continue
                else:
                    raise e
            
            # 验证阶段
            self.supernet.eval()
            val_loss = 0.0
            val_batches = 0
            
            # 创建验证批次进度条
            val_pbar = tqdm(
                val_loader,
                desc=f"{Fore.MAGENTA}  🔍 验证{Style.RESET_ALL}",
                leave=False,
                position=1,
                ncols=100,
                disable=len(val_loader) < 10  # 如果batch数量太少，禁用子进度条
            )
            
            with torch.no_grad():
                try:
                    for batch in val_pbar:
                        data, target = batch['image'].to(self.device), batch['label'].to(self.device)
                        
                        # 处理目标张量维度：移除多余的通道维度
                        if target.dim() == 5 and target.size(1) == 1:
                            target = target.squeeze(1)  # 从[B, 1, H, W, D]变为[B, H, W, D]
                        
                        # 确保目标张量为Long类型（交叉熵损失要求）
                        target = target.long()
                        
                        # 验证target张量值范围，防止CUDA设备端断言错误
                        target_min, target_max = target.min().item(), target.max().item()
                        if target_min < 0 or target_max >= self.num_classes:
                            logging.warning(f"验证阶段Target张量值超出范围: min={target_min}, max={target_max}, num_classes={self.num_classes}")
                            # 将target值限制在有效范围内
                            target = torch.clamp(target, 0, self.num_classes - 1)
                        
                        # 检查并处理NaN或无穷值
                        if torch.isnan(target).any() or torch.isinf(target).any():
                            logging.error("验证阶段Target张量包含NaN或无穷值，跳过此batch")
                            continue
                        
                        if self.use_amp:
                            with autocast('cuda'):
                                output = self.supernet(data)
                                batch_loss = F.cross_entropy(output, target).item()
                        else:
                            output = self.supernet(data)
                            batch_loss = F.cross_entropy(output, target).item()
                        
                        val_loss += batch_loss
                        val_batches += 1
                        
                        # 更新验证进度条
                        val_pbar.set_postfix({
                            'Loss': f'{batch_loss:.4f}',
                            'Avg': f'{val_loss/val_batches:.4f}'
                        })
                except RuntimeError as e:
                    if "DataLoader worker" in str(e):
                        logging.error(f"验证阶段DataLoader worker错误: {e}")
                        logging.info("尝试重新创建验证DataLoader...")
                        val_loader = self._recreate_dataloader(val_loader, num_workers=0)
                        continue
                    else:
                        raise e
            
            val_loss /= len(val_loader)
            epoch_time = time.time() - epoch_start_time
            
            # 更新主进度条
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss/train_batches:.4f}',
                'Val_Loss': f'{val_loss:.4f}',
                'Best': f'{best_val_loss:.4f}',
                'Time': f'{epoch_time:.1f}s',
                'ETA': self._estimate_remaining_time(epoch + 1, self.nas_epochs, search_start_time)
            })
            
            # 保存最优架构
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_arch_params = {name: param.clone() for name, param in self.supernet.arch_parameters.items()}
                
                # 显示新的最佳结果
                tqdm.write(f"{Fore.GREEN}✨ 新的最佳验证损失: {val_loss:.4f} (Epoch {epoch+1}){Style.RESET_ALL}")
                
                # 保存最优架构
                arch_save_path = self.save_dir / 'best_architecture.pth'
                torch.save({
                    'arch_parameters': best_arch_params,
                    'val_loss': best_val_loss,
                    'epoch': epoch
                }, arch_save_path)
            
            # 每个epoch结束后清理内存缓存
            self._clear_memory_cache()
        
        # 关闭进度条
        epoch_pbar.close()
        
        # 搜索完成总结
        search_time = time.time() - search_start_time
        print(f"\n{Fore.GREEN}🎉 NAS架构搜索完成!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 搜索统计:{Style.RESET_ALL}")
        print(f"   • 总耗时: {self._get_elapsed_time_str(search_start_time)}")
        print(f"   • 最佳验证损失: {Fore.YELLOW}{best_val_loss:.4f}{Style.RESET_ALL}")
        print(f"   • 内存使用: {self._get_memory_info_str()}")
        print(f"   • 架构保存至: {self.save_dir / 'best_architecture.pth'}")
        
        # 保存搜索结果
        self.best_architecture = {
            'arch_parameters': best_arch_params,
            'val_loss': best_val_loss,
            'supernet_state': self.supernet.state_dict()
        }
        
        self.nas_completed = True
        
        return self.best_architecture
        
    def create_student_model(self) -> nn.Module:
        """
        基于搜索到的最优架构创建学生模型
        
        Returns:
            学生模型
        """
        if not self.nas_completed or not self.best_architecture:
            raise ValueError("必须先完成NAS搜索才能创建学生模型")
            
        logging.info("基于最优架构创建学生模型...")
        
        # 创建新的SuperNet实例作为学生模型
        student = SuperNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=8,  # 大幅减少base_channels以节省内存
            num_layers=3,  # 减少层数
            dataset_type=self.dataset_type
        )
        
        # 加载最优架构参数
        student.arch_parameters.load_state_dict(self.best_architecture['arch_parameters'])
        
        # 冻结架构参数，只训练模型参数
        for param in student.arch_parameters.parameters():
            param.requires_grad = False
            
        self.student_model = student.to(self.device)
        logging.info("学生模型创建完成")
        
        return self.student_model
        
    def initialize_distillation(self) -> None:
        """
        初始化知识蒸馏组件
        """
        if not self.teacher_ensemble:
            self.initialize_teacher_models()
            
        if not self.student_model:
            self.create_student_model()
            
        logging.info("初始化知识蒸馏组件...")
        
        # 创建多教师蒸馏模型
        teacher_model_list = list(self.teacher_ensemble.values())  # 从字典中提取模型对象列表
        self.distillation_model = MultiTeacherDistillation(
            teacher_models=teacher_model_list,
            student_model=self.student_model,
            device=self.device,
            temperature=self.temperature
        )
        
        self.distillation_model = self.distillation_model.to(self.device)
        logging.info("知识蒸馏组件初始化完成")
        
    def _recreate_dataloader(self, original_loader, num_workers=0):
        """
        重新创建DataLoader，强制设置num_workers=0和Windows兼容性设置
        
        Args:
            original_loader: 原始DataLoader
            num_workers: worker进程数，默认为0
            
        Returns:
            新的DataLoader
        """
        from monai.data import DataLoader as MonaiDataLoader
        
        new_loader = MonaiDataLoader(
            original_loader.dataset,
            batch_size=original_loader.batch_size,
            shuffle=hasattr(original_loader, 'shuffle') and original_loader.shuffle,
            num_workers=num_workers,
            pin_memory=False,  # 禁用pin_memory以避免问题
            persistent_workers=False,  # Windows兼容性设置
            multiprocessing_context=None  # 避免多进程问题
        )
        
        logging.info(f"重新创建DataLoader: num_workers={num_workers}, batch_size={original_loader.batch_size}")
        return new_loader
        
    def _monitor_and_cleanup_memory(self):
        """
        监控内存使用并在必要时自动清理
        """
        if not torch.cuda.is_available():
            return
            
        allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        
        # 如果内存使用超过阈值，执行自动清理
        if allocated > self.auto_cleanup_threshold:
            logging.warning(f"内存使用过高 ({allocated:.2f}GB > {self.auto_cleanup_threshold}GB)，执行自动清理...")
            
            # 强制垃圾回收
            for _ in range(5):
                gc.collect()
            
            # 清空CUDA缓存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 检查清理效果
            new_allocated = torch.cuda.memory_allocated() / 1024**3
            freed_memory = allocated - new_allocated
            
            if freed_memory > 0.1:  # 释放了超过100MB
                logging.info(f"自动清理完成，释放了 {freed_memory:.2f}GB 内存")
            else:
                logging.warning("自动清理效果有限，可能存在内存泄漏")
        
        # 定期记录内存状态
        logging.debug(f"内存监控: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
        
    def _emergency_memory_cleanup(self):
        """
        紧急内存清理，用于处理内存不足的情况
        """
        logging.warning("执行紧急内存清理...")
        
        # 多次强制垃圾回收
        for _ in range(10):
            gc.collect()
        
        if torch.cuda.is_available():
            # 清空所有CUDA缓存
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # 尝试释放未使用的内存池
            try:
                torch.cuda.memory._dump_snapshot("emergency_cleanup.pickle")
            except:
                pass
            
            # 重置内存统计
            torch.cuda.reset_peak_memory_stats()
            
            allocated = torch.cuda.memory_allocated() / 1024**3
            logging.info(f"紧急清理完成，当前内存使用: {allocated:.2f}GB")
        
    def distillation_training(self, train_loader, val_loader) -> Dict[str, float]:
        """
        执行知识蒸馏训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练结果统计
        """
        if not self.distillation_model:
            self.initialize_distillation()
        
        # 打印阶段标题
        teacher_names = ', '.join(self.teacher_models)
        self._print_phase_header(
            "知识蒸馏训练阶段",
            f"教师模型: {teacher_names} | 蒸馏轮数: {self.distillation_epochs} | 学习率: {self.distillation_lr} | 温度: {self.temperature}"
        )
        
        # 优化器
        optimizer = torch.optim.Adam(
            self.distillation_model.student.parameters(),
            lr=self.distillation_lr,
            weight_decay=1e-4
        )
        
        distillation_start_time = time.time()
        best_val_loss = float('inf')
        training_stats = {'train_losses': [], 'val_losses': []}
        
        # 创建主进度条
        epoch_pbar = tqdm(
            range(self.distillation_epochs),
            desc=f"{Fore.GREEN}🎓 知识蒸馏{Style.RESET_ALL}",
            ncols=120,
            leave=True,
            position=0,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.distillation_epochs
        )
        
        for epoch in epoch_pbar:
            epoch_start_time = time.time()
            # 训练阶段
            self.distillation_model.train()
            train_loss = 0.0
            train_batches = 0
            
            # 创建训练批次进度条
            train_pbar = tqdm(
                enumerate(train_loader),
                total=len(train_loader),
                desc=f"{Fore.BLUE}  📚 蒸馏训练{Style.RESET_ALL}",
                leave=False,
                position=1,
                ncols=100,
                disable=len(train_loader) < 10  # 如果batch数量太少，禁用子进度条
            )
            
            try:
                for batch_idx, batch in train_pbar:
                    data, target = batch['image'].to(self.device), batch['label'].to(self.device)
                    
                    # 处理目标张量维度：移除多余的通道维度
                    if target.dim() == 5 and target.size(1) == 1:
                        target = target.squeeze(1)  # 从[B, 1, H, W, D]变为[B, H, W, D]
                    
                    optimizer.zero_grad()
                    
                    # 使用混合精度训练
                    if self.use_amp:
                        with autocast('cuda'):
                            # 前向传播
                            student_output = self.distillation_model.student(data)
                            
                            # 计算蒸馏损失
                            distillation_loss = self.distillation_model.compute_distillation_loss(
                                data, target
                            )
                        
                        self.scaler.scale(distillation_loss).backward()
                        self.scaler.step(optimizer)
                        self.scaler.update()
                    else:
                        # 前向传播
                        student_output = self.distillation_model.student(data)
                        
                        # 计算蒸馏损失
                        distillation_loss = self.distillation_model.compute_distillation_loss(
                            data, target
                        )
                        
                        distillation_loss.backward()
                        optimizer.step()
                    
                    train_loss += distillation_loss.item()
                    train_batches += 1
                    
                    # 更新训练进度条
                    train_pbar.set_postfix({
                        'Loss': f'{distillation_loss.item():.4f}',
                        'Avg': f'{train_loss/train_batches:.4f}',
                        'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                        'Mem': self._get_memory_info_str().split('|')[0].strip()
                    })
                    
                    # 内存监控和自动清理
                    if batch_idx % self.memory_monitor_interval == 0:
                        self._monitor_and_cleanup_memory()
            except RuntimeError as e:
                if "DataLoader worker" in str(e):
                    logging.error(f"蒸馏训练DataLoader worker错误: {e}")
                    logging.info("尝试重新创建训练DataLoader...")
                    train_loader = self._recreate_dataloader(train_loader, num_workers=0)
                    continue
                else:
                    raise e
            
            # 验证阶段
            self.distillation_model.eval()
            val_loss = 0.0
            val_batches = 0
            
            # 创建验证批次进度条
            val_pbar = tqdm(
                val_loader,
                desc=f"{Fore.MAGENTA}  🔍 蒸馏验证{Style.RESET_ALL}",
                leave=False,
                position=1,
                ncols=100,
                disable=len(val_loader) < 10  # 如果batch数量太少，禁用子进度条
            )
            
            with torch.no_grad():
                try:
                    for batch in val_pbar:
                        data, target = batch['image'].to(self.device), batch['label'].to(self.device)
                        
                        # 处理目标张量维度：移除多余的通道维度
                        if target.dim() == 5 and target.size(1) == 1:
                            target = target.squeeze(1)  # 从[B, 1, H, W, D]变为[B, H, W, D]
                        
                        if self.use_amp:
                            with autocast('cuda'):
                                batch_loss = self.distillation_model.compute_distillation_loss(
                                    data, target
                                ).item()
                        else:
                            batch_loss = self.distillation_model.compute_distillation_loss(
                                data, target
                            ).item()
                        
                        val_loss += batch_loss
                        val_batches += 1
                        
                        # 更新验证进度条
                        val_pbar.set_postfix({
                            'Loss': f'{batch_loss:.4f}',
                            'Avg': f'{val_loss/val_batches:.4f}'
                        })
                except RuntimeError as e:
                    if "DataLoader worker" in str(e):
                        logging.error(f"蒸馏验证DataLoader worker错误: {e}")
                        logging.info("尝试重新创建验证DataLoader...")
                        val_loader = self._recreate_dataloader(val_loader, num_workers=0)
                        continue
                    else:
                        raise e
            
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            epoch_time = time.time() - epoch_start_time
            
            training_stats['train_losses'].append(train_loss)
            training_stats['val_losses'].append(val_loss)
            
            # 更新主进度条
            epoch_pbar.set_postfix({
                'Train_Loss': f'{train_loss:.4f}',
                'Val_Loss': f'{val_loss:.4f}',
                'Best': f'{best_val_loss:.4f}',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}',
                'Time': f'{epoch_time:.1f}s',
                'ETA': self._estimate_remaining_time(epoch + 1, self.distillation_epochs, distillation_start_time)
            })
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_stats['best_epoch'] = epoch
                
                # 显示新的最佳结果
                tqdm.write(f"{Fore.GREEN}✨ 新的最佳蒸馏损失: {val_loss:.4f} (Epoch {epoch+1}){Style.RESET_ALL}")
                
                # 保存最优学生模型
                student_save_path = self.save_dir / 'best_student_model.pth'
                torch.save({
                    'model_state_dict': self.student_model.state_dict(),
                    'arch_parameters': self.best_architecture['arch_parameters'],
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'training_config': {
                        'teacher_models': self.teacher_models,
                        'dataset_type': self.dataset_type,
                        'temperature': self.temperature,
                        'alpha': self.alpha
                    }
                }, student_save_path)
                
            scheduler.step()
            
            # 每个epoch结束后清理内存缓存
            self._clear_memory_cache()
        
        # 关闭进度条
        epoch_pbar.close()
        
        # 蒸馏训练完成总结
        distillation_time = time.time() - distillation_start_time
        print(f"\n{Fore.GREEN}🎉 知识蒸馏训练完成!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 蒸馏统计:{Style.RESET_ALL}")
        print(f"   • 总耗时: {self._get_elapsed_time_str(distillation_start_time)}")
        print(f"   • 最佳验证损失: {Fore.YELLOW}{best_val_loss:.4f}{Style.RESET_ALL}")
        print(f"   • 最佳轮次: {training_stats['best_epoch'] + 1}")
        print(f"   • 教师模型: {', '.join(self.teacher_models)}")
        print(f"   • 内存使用: {self._get_memory_info_str()}")
        print(f"   • 学生模型保存至: {self.save_dir / 'best_student_model.pth'}")
        
        training_stats['best_val_loss'] = best_val_loss
        
        # 保存训练统计
        stats_save_path = self.save_dir / 'training_stats.json'
        with open(stats_save_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
        
        return training_stats
        
    def full_training_pipeline(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        执行完整的两阶段训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            完整训练结果
        """
        # 初始化训练开始时间
        self.training_start_time = time.time()
        
        # 打印总体流程标题
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🚀 NAS-蒸馏集成完整训练流程{Style.RESET_ALL}")
        print(f"{Fore.WHITE}教师模型: {', '.join(self.teacher_models)}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}数据集类型: {self.dataset_type}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}NAS轮数: {self.nas_epochs} | 蒸馏轮数: {self.distillation_epochs}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}保存目录: {self.save_dir}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}\n")
        
        results = {}
        
        # 阶段1：教师模型预训练
        print(f"{Fore.MAGENTA}📍 阶段 1/3: 教师模型预训练{Style.RESET_ALL}")
        pretrain_results = self.pretrain_teacher_models(train_loader, val_loader)
        results['teacher_pretrain'] = pretrain_results
        
        # 阶段间隔
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}📍 阶段 2/3: NAS架构搜索{Style.RESET_ALL}")
        
        # 阶段2：NAS架构搜索
        nas_results = self.search_architecture(train_loader, val_loader)
        results['nas_search'] = nas_results
        
        # 阶段间隔
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.MAGENTA}📍 阶段 3/3: 知识蒸馏训练{Style.RESET_ALL}")
        
        # 阶段3：知识蒸馏训练
        distillation_results = self.distillation_training(train_loader, val_loader)
        results['distillation'] = distillation_results
        
        # 完整流程总结
        total_time = time.time() - self.training_start_time
        print(f"\n{Fore.GREEN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}🎉 NAS-蒸馏集成完整训练流程完成!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 总体统计:{Style.RESET_ALL}")
        print(f"   • 总训练时间: {self._get_elapsed_time_str(self.training_start_time)}")
        print(f"   • 教师模型数量: {len(pretrain_results['pretrained_models'])}")
        print(f"   • NAS最佳验证损失: {Fore.YELLOW}{nas_results['val_loss']:.4f}{Style.RESET_ALL}")
        print(f"   • 蒸馏最佳验证损失: {Fore.YELLOW}{distillation_results['best_val_loss']:.4f}{Style.RESET_ALL}")
        print(f"   • 最终内存使用: {self._get_memory_info_str()}")
        print(f"   • 结果保存目录: {self.save_dir}")
        print(f"{Fore.GREEN}{'='*100}{Style.RESET_ALL}\n")
        
        # 保存完整结果
        results_save_path = self.save_dir / 'full_training_results.json'
        with open(results_save_path, 'w') as f:
            # 转换tensor为可序列化格式
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
        
        return results
        
    def load_trained_model(self, model_path: str) -> nn.Module:
        """
        加载训练好的学生模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的学生模型
        """
        logging.info(f"加载训练好的学生模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建学生模型
        student = SuperNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=8,  # 大幅减少base_channels以节省内存
            num_layers=3,  # 减少层数
            dataset_type=self.dataset_type
        )
        
        # 加载架构参数
        student.arch_parameters.load_state_dict(checkpoint['arch_parameters'])
        
        # 加载模型权重
        student.load_state_dict(checkpoint['model_state_dict'])
        
        student = student.to(self.device)
        student.eval()
        
        logging.info("学生模型加载完成")
        
        return student
        
    def _make_serializable(self, obj: Any) -> Any:
        """
        将包含tensor的对象转换为可序列化格式
        """
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
    
    def _print_phase_header(self, phase_name: str, details: str = ""):
        """打印训练阶段标题"""
        self.current_phase = phase_name
        self.phase_start_time = time.time()
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🚀 {phase_name}{Style.RESET_ALL}")
        if details:
            print(f"{Fore.WHITE}{details}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")
    
    def _get_memory_info_str(self) -> str:
        """获取内存信息字符串"""
        memory_stats = self.get_memory_stats()
        if torch.cuda.is_available():
            return f"GPU: {memory_stats['allocated_gb']:.1f}GB/{memory_stats['total_gpu_memory_gb']:.1f}GB | 利用率: {memory_stats['memory_utilization_percent']:.1f}%"
        else:
            # CPU模式下获取系统内存信息
            try:
                import psutil
                memory = psutil.virtual_memory()
                return f"系统内存: {memory.used / 1024**3:.1f}GB/{memory.total / 1024**3:.1f}GB | 利用率: {memory.percent:.1f}%"
            except ImportError:
                return "内存信息不可用"
    
    def _get_elapsed_time_str(self, start_time: float) -> str:
        """获取已用时间字符串"""
        elapsed = time.time() - start_time
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def _estimate_remaining_time(self, current_step: int, total_steps: int, start_time: float) -> str:
        """估算剩余时间"""
        if current_step == 0:
            return "--:--:--"
        
        elapsed = time.time() - start_time
        avg_time_per_step = elapsed / current_step
        remaining_steps = total_steps - current_step
        remaining_time = remaining_steps * avg_time_per_step
        
        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_memory_stats(self) -> Dict[str, float]:
        """
        获取详细的内存使用统计
        
        Returns:
            内存统计字典
        """
        stats = {}
        
        if torch.cuda.is_available():
            stats.update({
                'allocated_gb': torch.cuda.memory_allocated() / 1024**3,
                'reserved_gb': torch.cuda.memory_reserved() / 1024**3,
                'max_allocated_gb': torch.cuda.max_memory_allocated() / 1024**3,
                'max_reserved_gb': torch.cuda.max_memory_reserved() / 1024**3,
                'total_gpu_memory_gb': torch.cuda.get_device_properties(0).total_memory / 1024**3,
                'memory_utilization_percent': (torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory) * 100
            })
        else:
            stats = {
                'allocated_gb': 0.0,
                'reserved_gb': 0.0,
                'max_allocated_gb': 0.0,
                'max_reserved_gb': 0.0,
                'total_gpu_memory_gb': 0.0,
                'memory_utilization_percent': 0.0
            }
            
        return stats
            
    def get_save_paths(self) -> Dict[str, str]:
        """
        获取所有保存路径
        
        Returns:
            保存路径字典
        """
        return {
            'best_architecture': str(self.save_dir / 'best_architecture.pth'),
            'best_student_model': str(self.save_dir / 'best_student_model.pth'),
            'training_stats': str(self.save_dir / 'training_stats.json'),
            'full_results': str(self.save_dir / 'full_training_results.json'),
            'save_directory': str(self.save_dir)
        }


if __name__ == "__main__":
    # 测试代码
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # 创建模拟数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟BraTS数据
    batch_size = 2
    data = torch.randn(batch_size * 10, 4, 32, 32, 32)
    targets = torch.randint(0, 4, (batch_size * 10, 16, 16, 16))  # 匹配模型输出尺寸
    
    dataset = TensorDataset(data, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 创建NAS-蒸馏集成器
    teacher_models = ['UNet3D', 'VNet3D', 'ResUNet3D']
    
    nas_distillation = NASDistillationIntegration(
        teacher_models=teacher_models,
        device=device,
        dataset_type='BraTS',
        nas_epochs=5,  # 测试用较少轮数
        distillation_epochs=5,
        save_dir='./test_checkpoints/nas_distillation'
    )
    
    try:
        # 执行完整训练流程
        results = nas_distillation.full_training_pipeline(train_loader, val_loader)
        print("NAS-蒸馏集成测试成功！")
        print(f"保存路径: {nas_distillation.get_save_paths()}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()