import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
import time
import gc
from contextlib import contextmanager

from DatasetLoader_transforms import DatasetLoader
from model import BasicModelBank
from utils import VisualizationUtils, calculate_model_size
from monai.metrics import (
    DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric,
    ConfusionMatrixMetric, MeanIoU, GeneralizedDiceScore
)
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference

@contextmanager
def memory_efficient_context():
    """内存高效的上下文管理器"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class BraTSEvaluator:
    """
    脑肿瘤分割模型评估器 - 优化版本
    
    主要改进:
    - 统一的模型接口调用
    - 更好的内存管理
    - 增强的错误处理
    - 性能监控
    """
    def __init__(self, 
                 model_path: str,
                 data_dir: str,
                 device: str = "cuda",
                 output_dir: str = "./evaluation_results",
                 roi_size: Tuple[int, int, int] = (96, 96, 96),
                 sw_batch_size: int = 4,
                 overlap: float = 0.5):
        """
        初始化评估器
        
        Args:
            model_path: 模型检查点路径或目录路径
            data_dir: 数据目录
            device: 计算设备
            output_dir: 输出目录
            roi_size: 滑动窗口大小
            sw_batch_size: 滑动窗口批次大小
            overlap: 滑动窗口重叠率
        """
        # 解析模型路径，优先查找best_model.pth
        self.model_path = self._resolve_model_path(model_path)
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 推理参数
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        
        # 初始化模型类型标志
        self.is_advanced = False
        self.model_type = None
        
        # 设置随机种子
        set_determinism(seed=42)
        
        # 初始化组件
        self._load_model()
        self._setup_data()
        self._setup_metrics()
    def _resolve_model_path(self, model_path: str) -> str:
        """解析模型路径，优先查找best_model.pth"""
        if os.path.isfile(model_path):
            return model_path
        elif os.path.isdir(model_path):
            # 在目录中查找best_model.pth
            best_model_path = os.path.join(model_path, 'best_model.pth')
            if os.path.exists(best_model_path):
                print(f"找到最佳模型: {best_model_path}")
                return best_model_path
            else:
                raise FileNotFoundError(f"在目录 {model_path} 中未找到 best_model.pth")
        else:
            raise FileNotFoundError(f"模型路径不存在: {model_path}")
    
    def _load_model(self):
        """
        加载训练好的模型 - 支持四种模型类型：nas_model、distillation_student、fusion_model、basic_model
        从模型路径推断模型类型
        """
        print(f"加载模型: {self.model_path}")
        
        try:
            # 从模型路径推断模型类型
            model_path = Path(self.model_path)
            # 路径格式: ./outputs/models/{model_type}/checkpoints/best_model.pth
            # 获取模型类型目录名
            if 'outputs' in model_path.parts and 'models' in model_path.parts:
                models_index = model_path.parts.index('models')
                if models_index + 1 < len(model_path.parts):
                    model_type = model_path.parts[models_index + 1]
                else:
                    model_type = 'basic_model'
            else:
                # 如果路径不符合预期格式，尝试从父目录推断
                parent_dirs = [p.name for p in model_path.parents]
                supported_types = ['nas_model', 'distillation_student', 'fusion_model', 'basic_model']
                model_type = 'basic_model'  # 默认值
                for dir_name in parent_dirs:
                    if dir_name in supported_types:
                        model_type = dir_name
                        break
            
            # 支持的模型类型验证
            supported_types = ['nas_model', 'distillation_student', 'fusion_model', 'basic_model']
            if model_type not in supported_types:
                print(f"警告: 从路径推断的模型类型 {model_type} 不受支持，将作为基础模型处理")
                model_type = 'basic_model'
            
            self.model_type = model_type
            print(f"从路径推断的模型类型: {model_type}")
            
            # 加载检查点
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            config = checkpoint.get('config', {})
            
            # 获取模型名称（优先从配置中获取，否则使用默认值）
            model_name = config.get('model_name', 'UNet')
            print(f"模型名称: {model_name}")
            
            # 统一的模型加载逻辑
            self._load_unified_model(model_name, model_type, checkpoint)
                
            # 验证模型加载
            self._validate_model()
            
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载模型 {self.model_path}: {e}")
    
    def _load_unified_model(self, model_name: str, model_type: str, checkpoint: Dict):
        """
        统一的模型加载方法 - 支持四种模型类型：nas_model、distillation_student、fusion_model、basic_model
        所有模型都通过 BasicModelBank 加载
        """
        print(f"加载模型类型: {model_type}")
        
        try:
            # 统一使用 BasicModelBank 加载所有模型类型
            self.model_creator = BasicModelBank(
                model_name=model_name,
                device=self.device
            )
            
            # 加载模型权重
            if 'model_state_dict' in checkpoint:
                self.model_creator.model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                self.model_creator.model.load_state_dict(checkpoint['state_dict'])
            else:
                # 直接加载检查点作为状态字典
                self.model_creator.model.load_state_dict(checkpoint)
                
            self.model = self.model_creator.get_model()
            self.is_advanced = False
            print(f"成功加载 {model_type} 模型: {model_name}")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载模型 {model_name} (类型: {model_type}): {e}")
    

    
    def _validate_model(self):
        """验证模型加载是否成功"""
        if not hasattr(self, 'model_creator') or self.model_creator is None:
            raise RuntimeError("模型创建器未初始化")
        
        if not hasattr(self, 'model') or self.model is None:
            raise RuntimeError("模型未加载")
        
        # 设置模型为评估模式
        self.model.eval()
        
        print("模型验证通过，已设置为评估模式")
    
    def _unified_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """统一的推理接口 - 支持四种模型类型"""
        with memory_efficient_context():
            return self._basic_inference(inputs)
    
    def _basic_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """基础模型推理"""
        if hasattr(self.model_creator, 'sliding_window_inference'):
            return self.model_creator.sliding_window_inference(inputs)
        else:
            return sliding_window_inference(
                inputs=inputs,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self.model,
                overlap=self.overlap,
                mode='gaussian',
                sigma_scale=0.125,
                padding_mode='constant',
                cval=0.0
            )
        
    def _setup_data(self):
        """
        设置数据加载器
        """
        print("设置数据加载器...")
        
        data_loader = DatasetLoader(
            data_dir=self.data_dir,
            cache_rate=0.0,  # 评估时不使用缓存
            num_workers=0,  # Windows上设置为0避免多进程问题
            seed=42
        )
        
        # 获取验证数据
        _, self.val_loader = data_loader.get_dataloaders(batch_size=1)  # 评估时使用batch_size=1
        
        print(f"验证样本数: {len(self.val_loader)}")
        
    def _setup_metrics(self):
        """
        设置评估指标
        """
        # 基础分割指标
        self.dice_metric = DiceMetric(
            include_background=False, 
            reduction="mean_batch",
            get_not_nans=False
        )
        
        self.hd_metric = HausdorffDistanceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False
        )
        
        self.surface_metric = SurfaceDistanceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False
        )
        
        # 启用所有评估指标
        print("启用所有评估指标")
        from monai.metrics import ConfusionMatrixMetric, MeanIoU, GeneralizedDiceScore
        
        self.confusion_matrix_metric = ConfusionMatrixMetric(
            include_background=False,
            metric_name="confusion_matrix",
            compute_sample=True,
            reduction="mean_batch"
        )
        
        self.iou_metric = MeanIoU(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False
        )
        
        self.generalized_dice_metric = GeneralizedDiceScore(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False
        )
        
        self.advanced_metrics_enabled = True
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=4)])
        self.post_label = Compose([AsDiscrete(to_onehot=4)])
    def evaluate_model(self) -> Dict[str, float]:
        """
        评估模型性能
        """
        print("开始模型评估...")
        
        # 重置指标
        self.dice_metric.reset()
        self.hd_metric.reset()
        self.surface_metric.reset()
        
        if self.advanced_metrics_enabled:
            self.confusion_matrix_metric.reset()
            self.iou_metric.reset()
            self.generalized_dice_metric.reset()
        
        all_dice_scores = []
        all_hd_scores = []
        all_surface_scores = []
        all_iou_scores = []
        all_generalized_dice_scores = []
        
        case_results = []
        
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(tqdm(self.val_loader, desc="评估进度")):
                inputs = batch_data['image'].to(self.device)
                labels = batch_data['label'].to(self.device)
                subject_id = batch_data.get('subject_id', [f'case_{batch_idx}'])[0]
                
                # 统一推理接口
                start_time = time.time()
                outputs = self._unified_inference(inputs)
                inference_time = time.time() - start_time
                
                # 后处理
                outputs_list = decollate_batch(outputs)
                labels_list = decollate_batch(labels)
                
                outputs_convert = [self.post_pred(pred) for pred in outputs_list]
                labels_convert = [self.post_label(label) for label in labels_list]
                
                # 计算基础指标
                dice_scores = self.dice_metric(y_pred=outputs_convert, y=labels_convert)
                hd_scores = self.hd_metric(y_pred=outputs_convert, y=labels_convert)
                surface_scores = self.surface_metric(y_pred=outputs_convert, y=labels_convert)
                
                # 存储单个案例结果
                case_dice = dice_scores.mean().item()
                case_hd = hd_scores.mean().item() if not torch.isnan(hd_scores.mean()) else float('inf')
                case_surface = surface_scores.mean().item() if not torch.isnan(surface_scores.mean()) else float('inf')
                
                all_dice_scores.append(case_dice)
                all_hd_scores.append(case_hd)
                all_surface_scores.append(case_surface)
                
                case_result = {
                    'subject_id': subject_id,
                    'dice': case_dice,
                    'hausdorff_distance': case_hd,
                    'surface_distance': case_surface,
                    'inference_time': inference_time
                }
                
                # 计算高级指标（如果启用）
                if self.advanced_metrics_enabled:
                    iou_scores = self.iou_metric(y_pred=outputs_convert, y=labels_convert)
                    generalized_dice_scores = self.generalized_dice_metric(y_pred=outputs_convert, y=labels_convert)
                    confusion_matrix = self.confusion_matrix_metric(y_pred=outputs_convert, y=labels_convert)
                    
                    case_iou = iou_scores.mean().item() if not torch.isnan(iou_scores.mean()) else 0.0
                    case_generalized_dice = generalized_dice_scores.mean().item() if not torch.isnan(generalized_dice_scores.mean()) else 0.0
                    
                    # 处理混淆矩阵结果
                    confusion_matrix_np = confusion_matrix.cpu().numpy() if hasattr(confusion_matrix, 'cpu') else confusion_matrix
                    
                    all_iou_scores.append(case_iou)
                    all_generalized_dice_scores.append(case_generalized_dice)
                    
                    case_result.update({
                        'iou': case_iou,
                        'generalized_dice': case_generalized_dice,
                        'confusion_matrix': confusion_matrix_np.tolist()  # 转换为列表以便JSON序列化
                    })
                
                case_results.append(case_result)
                
                # 保存部分可视化结果
                if batch_idx < 5:  # 只保存前5个案例的可视化
                    self._save_visualization(
                        inputs[0].cpu().numpy(),
                        labels[0].cpu().numpy(),
                        torch.argmax(outputs[0], dim=0).cpu().numpy(),
                        subject_id,
                        case_dice
                    )
        
        # 计算总体统计
        results = {
            'mean_dice': np.mean(all_dice_scores),
            'std_dice': np.std(all_dice_scores),
            'median_dice': np.median(all_dice_scores),
            'min_dice': np.min(all_dice_scores),
            'max_dice': np.max(all_dice_scores),
            
            'mean_hd': np.mean([hd for hd in all_hd_scores if hd != float('inf')]),
            'std_hd': np.std([hd for hd in all_hd_scores if hd != float('inf')]),
            'median_hd': np.median([hd for hd in all_hd_scores if hd != float('inf')]),
            
            'mean_surface': np.mean([sd for sd in all_surface_scores if sd != float('inf')]),
            'std_surface': np.std([sd for sd in all_surface_scores if sd != float('inf')]),
            'median_surface': np.median([sd for sd in all_surface_scores if sd != float('inf')]),
            
            'total_cases': len(all_dice_scores)
        }

        if self.advanced_metrics_enabled and all_iou_scores:
            results.update({
                'mean_iou': np.mean(all_iou_scores),
                'std_iou': np.std(all_iou_scores),
                'median_iou': np.median(all_iou_scores),
                'min_iou': np.min(all_iou_scores),
                'max_iou': np.max(all_iou_scores),
                
                'mean_generalized_dice': np.mean(all_generalized_dice_scores),
                'std_generalized_dice': np.std(all_generalized_dice_scores),
                'median_generalized_dice': np.median(all_generalized_dice_scores),
                'min_generalized_dice': np.min(all_generalized_dice_scores),
                'max_generalized_dice': np.max(all_generalized_dice_scores)
            })
        
        # 保存详细结果
        self._save_detailed_results(case_results, results)
        
        return results
    
    def _save_visualization(self, 
                          images: np.ndarray,
                          labels: np.ndarray, 
                          predictions: np.ndarray,
                          subject_id: str,
                          dice_score: float):
        """
        保存可视化结果
        images: 输入图像
        labels: 真实标签
        predictions: 预测结果
        subject_id: 病例ID
        dice_score: Dice分数
        """
        # 选择中间切片
        slice_idx = images.shape[-1] // 2
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f'{subject_id} - Dice: {dice_score:.4f}', fontsize=16)
        
        # 显示不同模态的图像
        modalities = ['T1n', 'T1c', 'T2w', 'T2f']
        for i in range(min(4, images.shape[0])):
            row = i // 2
            col = i % 2
            if row < 2 and col < 2:
                axes[row, col].imshow(images[i, :, :, slice_idx], cmap='gray')
                axes[row, col].set_title(f'{modalities[i]}')
                axes[row, col].axis('off')
        
        # 显示真实标签
        axes[0, 2].imshow(images[0, :, :, slice_idx], cmap='gray')
        axes[0, 2].imshow(labels[:, :, slice_idx], cmap='jet', alpha=0.5)
        axes[0, 2].set_title('真实标签')
        axes[0, 2].axis('off')
        
        # 显示预测结果
        axes[1, 2].imshow(images[0, :, :, slice_idx], cmap='gray')
        axes[1, 2].imshow(predictions[:, :, slice_idx], cmap='jet', alpha=0.5)
        axes[1, 2].set_title('预测结果')
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # 保存图像
        save_path = self.output_dir / 'visualizations' / f'{subject_id}_segmentation.png'
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _save_detailed_results(self, case_results: List[Dict], summary_results: Dict[str, float]):
        """
        保存详细评估结果
        case_results: 每个案例的结果
        summary_results: 总体统计结果
        """
        # 保存案例级别结果
        df_cases = pd.DataFrame(case_results)
        df_cases.to_csv(self.output_dir / 'case_results.csv', index=False)
        
        # 保存总体统计
        with open(self.output_dir / 'summary_results.txt', 'w', encoding='utf-8') as f:
            f.write("BraTS脑肿瘤分割模型评估结果\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("Dice系数统计:\n")
            f.write(f"  平均值: {summary_results['mean_dice']:.4f} ± {summary_results['std_dice']:.4f}\n")
            f.write(f"  中位数: {summary_results['median_dice']:.4f}\n")
            f.write(f"  最小值: {summary_results['min_dice']:.4f}\n")
            f.write(f"  最大值: {summary_results['max_dice']:.4f}\n\n")
            
            f.write("Hausdorff距离统计:\n")
            f.write(f"  平均值: {summary_results['mean_hd']:.4f} ± {summary_results['std_hd']:.4f}\n")
            f.write(f"  中位数: {summary_results['median_hd']:.4f}\n\n")
            
            f.write("表面距离统计:\n")
            f.write(f"  平均值: {summary_results['mean_surface']:.4f} ± {summary_results['std_surface']:.4f}\n")
            f.write(f"  中位数: {summary_results['median_surface']:.4f}\n\n")
            
            f.write(f"总案例数: {summary_results['total_cases']}\n")
        
        # 绘制结果分布图
        self._plot_results_distribution(case_results)
        
        print(f"详细结果已保存到: {self.output_dir}")
    
    def _plot_results_distribution(self, case_results: List[Dict]):
        """
        绘制结果分布图
        case_results: 案例结果列表
        """
        df = pd.DataFrame(case_results)
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('评估指标分布', fontsize=16)
        
        # Dice分数分布
        axes[0, 0].hist(df['dice'], bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].axvline(df['dice'].mean(), color='red', linestyle='--', 
                          label=f'平均值: {df["dice"].mean():.4f}')
        axes[0, 0].set_title('Dice系数分布')
        axes[0, 0].set_xlabel('Dice Score')
        axes[0, 0].set_ylabel('频次')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Hausdorff距离分布
        hd_valid = df[df['hausdorff_distance'] != float('inf')]['hausdorff_distance']
        if len(hd_valid) > 0:
            axes[0, 1].hist(hd_valid, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].axvline(hd_valid.mean(), color='red', linestyle='--',
                              label=f'平均值: {hd_valid.mean():.4f}')
            axes[0, 1].set_title('Hausdorff距离分布')
            axes[0, 1].set_xlabel('Hausdorff Distance')
            axes[0, 1].set_ylabel('频次')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Dice分数箱线图
        axes[1, 0].boxplot(df['dice'], vert=True)
        axes[1, 0].set_title('Dice系数箱线图')
        axes[1, 0].set_ylabel('Dice Score')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 散点图：Dice vs Hausdorff
        hd_for_scatter = df['hausdorff_distance'].replace(float('inf'), np.nan)
        axes[1, 1].scatter(df['dice'], hd_for_scatter, alpha=0.6)
        axes[1, 1].set_title('Dice vs Hausdorff距离')
        axes[1, 1].set_xlabel('Dice Score')
        axes[1, 1].set_ylabel('Hausdorff Distance')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'results_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_models(self, model_paths: List[str], model_names: List[str]) -> pd.DataFrame:
        """
        比较多个模型的性能
        model_names: 模型名称列表
        比较结果DataFrame
        """
        comparison_results = []
        
        for model_path, model_name in zip(model_paths, model_names):
            print(f"\n评估模型: {model_name}")
            
            # 临时更新模型路径
            original_path = self.model_path
            self.model_path = model_path
            
            # 重新加载模型
            self._load_model()
            
            # 评估模型
            results = self.evaluate_model()
            results['model_name'] = model_name
            comparison_results.append(results)
            
            # 恢复原始路径
            self.model_path = original_path
        
        # 创建比较表格
        df_comparison = pd.DataFrame(comparison_results)
        df_comparison = df_comparison[['model_name', 'mean_dice', 'std_dice', 'mean_hd', 'std_hd']]
        
        # 保存比较结果
        df_comparison.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        print("\n模型比较结果:")
        print(df_comparison.to_string(index=False))
        
        return df_comparison

def main():
    pass

if __name__ == '__main__':
    main()

# 注意: 此文件主要通过 main.py 调用，不提供独立的命令行接口
# 如需独立使用评估功能，请使用: python main.py --mode evaluate