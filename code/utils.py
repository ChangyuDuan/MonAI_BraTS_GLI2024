import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

class EarlyStopping:
    """
    早停机制实现
    """
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001, mode: str = 'min'):
        """
        初始化早停机制
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善阈值
            mode: 监控模式 ('min' 或 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        检查是否应该早停
        
        Args:
            score: 当前监控指标值
            
        Returns:
            是否应该早停
        """
        if self.best_score is None:
            self.best_score = score
        elif self._is_improvement(score):
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
                
        return self.early_stop
    
    def _is_improvement(self, score: float) -> bool:
        """
        判断是否有改善
        """
        if self.mode == 'min':
            return score < self.best_score - self.min_delta
        else:
            return score > self.best_score + self.min_delta

class ModelCheckpoint:
    """
    模型检查点保存
    """
    
    def __init__(self, save_dir: str, filename: str = 'best_model.pth'):
        """
        初始化模型检查点
        
        Args:
            save_dir: 保存目录
            filename: 文件名
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.filepath = self.save_dir / filename
        
    def save(self, state_dict: Dict[str, Any]):
        """
        保存模型检查点
        
        Args:
            state_dict: 状态字典
        """
        # 添加保存时间戳
        state_dict['save_time'] = datetime.now().isoformat()
        
        torch.save(state_dict, self.filepath)
        
        # 同时保存一个带时间戳的副本
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"model_{timestamp}.pth"
        backup_filepath = self.save_dir / backup_filename
        torch.save(state_dict, backup_filepath)
        
    def load(self, device: str = 'cpu') -> Dict[str, Any]:
        """
        加载模型检查点
        
        Args:
            device: 加载到的设备
            
        Returns:
            状态字典
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"检查点文件不存在: {self.filepath}")
            
        return torch.load(self.filepath, map_location=device)

class MetricsTracker:
    """
    训练指标跟踪器
    """
    
    def __init__(self):
        """
        初始化指标跟踪器
        """
        self.history = defaultdict(list)
        
    def update(self, metrics: Dict[str, float]):
        """
        更新指标
        
        Args:
            metrics: 指标字典
        """
        for key, value in metrics.items():
            self.history[key].append(value)
            
    def get_history(self) -> Dict[str, List[float]]:
        """
        获取历史记录
        
        Returns:
            历史记录字典
        """
        return dict(self.history)
    
    def get_latest(self) -> Dict[str, float]:
        """
        获取最新指标
        
        Returns:
            最新指标字典
        """
        return {key: values[-1] for key, values in self.history.items() if values}
    
    def get_best(self, metric_name: str, mode: str = 'max') -> float:
        """
        获取最佳指标值
        
        Args:
            metric_name: 指标名称
            mode: 'max' 或 'min'
            
        Returns:
            最佳指标值
        """
        if metric_name not in self.history:
            raise KeyError(f"指标 '{metric_name}' 不存在")
            
        values = self.history[metric_name]
        return max(values) if mode == 'max' else min(values)
    
    def save_history(self, filepath: str):
        """
        保存历史记录到文件
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dict(self.history), f, indent=2, ensure_ascii=False)
    
    def load_history(self, filepath: str):
        """
        从文件加载历史记录
        
        Args:
            filepath: 文件路径
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            loaded_history = json.load(f)
            self.history = defaultdict(list, loaded_history)

class VisualizationUtils:
    """
    可视化工具类
    """
    
    @staticmethod
    def plot_training_metrics(history: Dict[str, List[float]], 
                            save_path: Optional[str] = None,
                            figsize: tuple = (15, 10)):
        """
        绘制训练指标
        
        Args:
            history: 训练历史
            save_path: 保存路径
            figsize: 图像大小
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('训练过程监控', fontsize=16)
        
        # 损失曲线
        if 'loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['loss'], label='训练损失', color='blue')
            axes[0, 0].plot(history['val_loss'], label='验证损失', color='red')
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Dice系数曲线
        if 'dice' in history and 'val_dice' in history:
            axes[0, 1].plot(history['dice'], label='训练Dice', color='blue')
            axes[0, 1].plot(history['val_dice'], label='验证Dice', color='red')
            axes[0, 1].set_title('Dice系数曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 学习率曲线
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], label='学习率', color='green')
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].set_yscale('log')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # 验证指标对比
        if 'val_loss' in history and 'val_dice' in history:
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
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_segmentation_results(images: np.ndarray, 
                                labels: np.ndarray, 
                                predictions: np.ndarray,
                                slice_idx: int = None,
                                save_path: Optional[str] = None):
        """
        可视化分割结果
        
        Args:
            images: 输入图像 (C, H, W, D)
            labels: 真实标签 (H, W, D)
            predictions: 预测结果 (H, W, D)
            slice_idx: 切片索引
            save_path: 保存路径
        """
        if slice_idx is None:
            slice_idx = images.shape[-1] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'分割结果可视化 (切片 {slice_idx})', fontsize=16)
        
        # 显示不同模态的图像
        modalities = ['T1n', 'T1c', 'T2w', 'T2f']
        for i in range(min(4, images.shape[0])):
            row = i // 2
            col = i % 2
            if row < 2 and col < 2:
                axes[row, col].imshow(images[i, :, :, slice_idx], cmap='gray')
                axes[row, col].set_title(f'{modalities[i]} 图像')
                axes[row, col].axis('off')
        
        # 显示真实标签
        axes[0, 2].imshow(labels[:, :, slice_idx], cmap='jet', alpha=0.7)
        axes[0, 2].set_title('真实标签')
        axes[0, 2].axis('off')
        
        # 显示预测结果
        axes[1, 2].imshow(predictions[:, :, slice_idx], cmap='jet', alpha=0.7)
        axes[1, 2].set_title('预测结果')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"分割结果已保存: {save_path}")
        
        plt.show()



def calculate_model_size(model: torch.nn.Module) -> Dict[str, Any]:
    """
    计算模型大小信息
    
    Args:
        model: PyTorch模型
        
    Returns:
        模型大小信息字典
    """
    param_count = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # 估算模型大小（MB）
    param_size = param_count * 4 / (1024 * 1024)  # 假设float32
    
    return {
        'total_params': param_count,
        'trainable_params': trainable_param_count,
        'non_trainable_params': param_count - trainable_param_count,
        'model_size_mb': param_size
    }

def format_time(seconds: float) -> str:
    """
    格式化时间显示
    
    Args:
        seconds: 秒数
        
    Returns:
        格式化的时间字符串
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{int(minutes)}m {seconds:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {seconds:.1f}s"

if __name__ == "__main__":
    # 测试工具函数
    print("测试BraTS项目工具函数...")
    print("=" * 50)
    
    # 测试早停机制（使用项目默认配置）
    print("\n1. 测试早停机制:")
    early_stopping = EarlyStopping(patience=30, min_delta=0.001)  # 使用项目默认配置
    scores = [1.0, 0.9, 0.85, 0.84, 0.83, 0.82, 0.81, 0.80]
    
    for i, score in enumerate(scores):
        should_stop = early_stopping(score)
        print(f"  Epoch {i+1}: Validation Loss={score:.3f}, Should stop={should_stop}")
        if should_stop:
            print(f"  早停触发！在第{i+1}轮停止训练")
            break
    
    # 测试指标跟踪器
    print("\n2. 测试指标跟踪器:")
    tracker = MetricsTracker()
    for epoch in range(10):
        # 模拟训练过程中的指标变化
        metrics = {
            'train_loss': 1.0 - epoch * 0.08,
            'val_loss': 1.2 - epoch * 0.07,
            'train_dice': 0.3 + epoch * 0.06,
            'val_dice': 0.25 + epoch * 0.055,
            'learning_rate': 2e-4 * (0.95 ** epoch)
        }
        tracker.update(metrics)
    
    print(f"  最新指标: {tracker.get_latest()}")
    print(f"  最佳验证Dice: {tracker.get_best('val_dice', 'max'):.4f}")
    print(f"  最低验证损失: {tracker.get_best('val_loss', 'min'):.4f}")

    
    # 测试时间格式化
    print("\n4. 测试时间格式化:")
    test_times = [30, 150, 3661, 7322]
    for t in test_times:
        print(f"  {t}秒 -> {format_time(t)}")
    
    print("\n" + "=" * 50)
    print("BraTS项目工具函数测试完成！")
    print("所有工具类已准备就绪，可用于训练和评估流程。")