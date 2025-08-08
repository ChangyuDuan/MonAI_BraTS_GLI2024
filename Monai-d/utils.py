import os
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import matplotlib.pyplot as plt
from datetime import datetime

# 导入中文字体配置
try:
    from font_config import configure_chinese_font
    # 自动配置中文字体
    configure_chinese_font()
except ImportError:
    import warnings
    warnings.warn("未找到font_config模块，中文显示可能出现问题", UserWarning)

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
    可视化工具类 - 支持所有7个评估指标的完整可视化
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
    def plot_all_metrics_distribution(case_results: List[Dict], 
                                    save_path: Optional[str] = None,
                                    figsize: tuple = (20, 15)):
        """
        绘制所有7个指标的分布图
        
        Args:
            case_results: 案例结果列表
            save_path: 保存路径
            figsize: 图像大小
        """
        import pandas as pd
        
        df = pd.DataFrame(case_results)
        
        # 定义7个指标及其显示名称
        metrics_info = {
            'dice': {'name': 'Dice系数', 'color': 'blue'},
            'hausdorff_distance': {'name': 'Hausdorff距离', 'color': 'green'},
            'surface_distance': {'name': '表面距离', 'color': 'red'},
            'iou': {'name': 'IoU', 'color': 'orange'},
            'generalized_dice': {'name': '广义Dice', 'color': 'purple'},
            'froc_auc': {'name': 'FROC AUC', 'color': 'brown'},
            'sensitivity': {'name': '敏感度', 'color': 'pink'}
        }
        
        # 创建子图
        fig, axes = plt.subplots(3, 3, figsize=figsize)
        fig.suptitle('所有评估指标分布图', fontsize=20)
        
        plot_idx = 0
        for metric_key, info in metrics_info.items():
            if metric_key in df.columns:
                row = plot_idx // 3
                col = plot_idx % 3
                
                # 处理无穷值
                data = df[metric_key]
                if metric_key in ['hausdorff_distance', 'surface_distance']:
                    data = data[data != float('inf')]
                
                if len(data) > 0:
                    # 直方图
                    axes[row, col].hist(data, bins=20, alpha=0.7, 
                                      color=info['color'], edgecolor='black')
                    axes[row, col].axvline(data.mean(), color='red', linestyle='--',
                                         label=f'均值: {data.mean():.4f}')
                    axes[row, col].set_title(f'{info["name"]}分布')
                    axes[row, col].set_xlabel(info['name'])
                    axes[row, col].set_ylabel('频次')
                    axes[row, col].legend()
                    axes[row, col].grid(True, alpha=0.3)
                
                plot_idx += 1
        
        # 如果有空余的子图，隐藏它们
        for i in range(plot_idx, 9):
            row = i // 3
            col = i % 3
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"指标分布图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_metrics_comparison(case_results: List[Dict], 
                              save_path: Optional[str] = None,
                              figsize: tuple = (15, 10)):
        """
        绘制指标对比箱线图
        
        Args:
            case_results: 案例结果列表
            save_path: 保存路径
            figsize: 图像大小
        """
        import pandas as pd
        
        df = pd.DataFrame(case_results)
        
        # 选择主要指标进行对比
        main_metrics = ['dice', 'iou', 'generalized_dice', 'sensitivity']
        available_metrics = [m for m in main_metrics if m in df.columns]
        
        if not available_metrics:
            print("警告: 没有找到可用的指标进行对比")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('主要指标对比分析', fontsize=16)
        
        # 箱线图
        data_for_box = [df[metric].dropna() for metric in available_metrics]
        axes[0].boxplot(data_for_box, labels=available_metrics)
        axes[0].set_title('指标箱线图对比')
        axes[0].set_ylabel('指标值')
        axes[0].grid(True, alpha=0.3)
        axes[0].tick_params(axis='x', rotation=45)
        
        # 小提琴图
        positions = range(1, len(available_metrics) + 1)
        for i, metric in enumerate(available_metrics):
            data = df[metric].dropna()
            if len(data) > 0:
                parts = axes[1].violinplot([data], positions=[positions[i]], 
                                         showmeans=True, showmedians=True)
                # 设置颜色
                colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink']
                for pc in parts['bodies']:
                    pc.set_facecolor(colors[i % len(colors)])
                    pc.set_alpha(0.7)
        
        axes[1].set_title('指标分布密度对比')
        axes[1].set_ylabel('指标值')
        axes[1].set_xticks(positions)
        axes[1].set_xticklabels(available_metrics, rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"指标对比图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_froc_curve(froc_data: Dict, 
                       save_path: Optional[str] = None,
                       figsize: tuple = (10, 8)):
        """
        绘制FROC曲线
        
        Args:
            froc_data: FROC数据字典，包含fp_rates和sensitivities
            save_path: 保存路径
            figsize: 图像大小
        """
        if 'fp_rates' not in froc_data or 'sensitivities' not in froc_data:
            print("警告: FROC数据不完整，无法绘制FROC曲线")
            return
        
        fig, ax = plt.subplots(figsize=figsize)
        
        fp_rates = froc_data['fp_rates']
        sensitivities = froc_data['sensitivities']
        auc = froc_data.get('auc', 0.0)
        
        # 绘制FROC曲线
        ax.plot(fp_rates, sensitivities, 'b-', linewidth=2, 
               label=f'FROC曲线 (AUC = {auc:.4f})')
        
        # 标记特定点
        target_fp_rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        for target_fp in target_fp_rates:
            if target_fp in fp_rates:
                idx = fp_rates.index(target_fp)
                sensitivity = sensitivities[idx]
                ax.plot(target_fp, sensitivity, 'ro', markersize=8)
                ax.annotate(f'({target_fp}, {sensitivity:.3f})', 
                          (target_fp, sensitivity), 
                          xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('平均假阳性数/图像')
        ax.set_ylabel('敏感度')
        ax.set_title('FROC曲线 (Free-Response Operating Characteristic)')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([0, max(fp_rates) * 1.1])
        ax.set_ylim([0, 1.05])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"FROC曲线已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_confusion_matrix_heatmap(confusion_matrices: List[np.ndarray], 
                                    class_names: List[str],
                                    save_path: Optional[str] = None,
                                    figsize: tuple = (12, 8)):
        """
        绘制混淆矩阵热力图
        
        Args:
            confusion_matrices: 混淆矩阵列表
            class_names: 类别名称列表
            save_path: 保存路径
            figsize: 图像大小
        """
        import seaborn as sns
        
        # 计算平均混淆矩阵
        avg_cm = np.mean(confusion_matrices, axis=0)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('混淆矩阵分析', fontsize=16)
        
        # 绝对数值热力图
        sns.heatmap(avg_cm, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[0])
        axes[0].set_title('平均混淆矩阵 (绝对值)')
        axes[0].set_xlabel('预测类别')
        axes[0].set_ylabel('真实类别')
        
        # 归一化热力图
        normalized_cm = avg_cm / avg_cm.sum(axis=1, keepdims=True)
        sns.heatmap(normalized_cm, annot=True, fmt='.3f', cmap='Reds',
                   xticklabels=class_names, yticklabels=class_names,
                   ax=axes[1])
        axes[1].set_title('归一化混淆矩阵 (比例)')
        axes[1].set_xlabel('预测类别')
        axes[1].set_ylabel('真实类别')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵热力图已保存: {save_path}")
        
        plt.show()
    
    @staticmethod
    def plot_metrics_correlation(case_results: List[Dict], 
                               save_path: Optional[str] = None,
                               figsize: tuple = (12, 10)):
        """
        绘制指标相关性分析
        
        Args:
            case_results: 案例结果列表
            save_path: 保存路径
            figsize: 图像大小
        """
        import pandas as pd
        import seaborn as sns
        
        df = pd.DataFrame(case_results)
        
        # 选择数值型指标
        numeric_metrics = ['dice', 'iou', 'generalized_dice', 'sensitivity']
        # 处理距离指标（移除无穷值）
        if 'hausdorff_distance' in df.columns:
            df['hausdorff_distance'] = df['hausdorff_distance'].replace(float('inf'), np.nan)
            numeric_metrics.append('hausdorff_distance')
        if 'surface_distance' in df.columns:
            df['surface_distance'] = df['surface_distance'].replace(float('inf'), np.nan)
            numeric_metrics.append('surface_distance')
        if 'froc_auc' in df.columns:
            numeric_metrics.append('froc_auc')
        
        # 过滤存在的指标
        available_metrics = [m for m in numeric_metrics if m in df.columns]
        
        if len(available_metrics) < 2:
            print("警告: 可用指标不足，无法进行相关性分析")
            return
        
        # 计算相关性矩阵
        correlation_matrix = df[available_metrics].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制相关性热力图
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   ax=ax)
        
        ax.set_title('评估指标相关性分析', fontsize=16)
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"指标相关性图已保存: {save_path}")
        
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
    print("测试医学图像分割项目工具函数...")
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
    print("医学图像分割项目工具函数测试完成！")
    print("所有工具类已准备就绪，可用于训练和评估流程。")

def visualize_prediction_3d(image: np.ndarray, 
                           prediction: np.ndarray, 
                           save_path: str,
                           slice_indices: Optional[List[int]] = None,
                           figsize: tuple = (15, 5)) -> None:
    """
    可视化3D医学图像的预测结果
    
    Args:
        image: 原始图像数组 (H, W, D) 或 (C, H, W, D)
        prediction: 预测结果数组 (H, W, D) 或 (C, H, W, D)
        save_path: 保存路径
        slice_indices: 要显示的切片索引列表，如果为None则自动选择中间切片
        figsize: 图像大小
    """
    import matplotlib.pyplot as plt
    
    # 确保输入是3D数组
    if image.ndim == 4:
        image = image[0]  # 取第一个通道
    if prediction.ndim == 4:
        prediction = prediction[0]  # 取第一个通道
    
    # 如果没有指定切片索引，选择中间切片
    if slice_indices is None:
        depth = image.shape[-1]
        slice_indices = [depth // 4, depth // 2, 3 * depth // 4]
    
    # 创建子图
    fig, axes = plt.subplots(2, len(slice_indices), figsize=figsize)
    if len(slice_indices) == 1:
        axes = axes.reshape(2, 1)
    
    for i, slice_idx in enumerate(slice_indices):
        # 确保切片索引在有效范围内
        slice_idx = max(0, min(slice_idx, image.shape[-1] - 1))
        
        # 显示原始图像
        axes[0, i].imshow(image[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'原始图像 - 切片 {slice_idx}')
        axes[0, i].axis('off')
        
        # 显示预测结果
        axes[1, i].imshow(prediction[:, :, slice_idx], cmap='jet', alpha=0.7)
        axes[1, i].imshow(image[:, :, slice_idx], cmap='gray', alpha=0.3)
        axes[1, i].set_title(f'预测结果 - 切片 {slice_idx}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到: {save_path}")