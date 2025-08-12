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
from MSMultiSpineLoader import MSMultiSpineDatasetLoader, create_optimized_training_config
from model import BasicModelBank
from utils import VisualizationUtils, calculate_model_size
from monai.metrics import (
    DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric,
    ConfusionMatrixMetric, MeanIoU, GeneralizedDiceScore
)
from scipy import ndimage
from sklearn.metrics import roc_curve, auc
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
from monai.utils import set_determinism
from monai.inferers import sliding_window_inference
from model import SpecializedModelFactory

# 导入中文字体配置（不自动调用，由main.py统一配置）
try:
    from font_config import configure_chinese_font
except ImportError:
    import warnings
    warnings.warn("未找到font_config模块，中文显示可能出现问题", UserWarning)

class ImprovedFROCEvaluator:
    """
    改进的FROC曲线评估器
    专为真实训练模型设计，提供准确、可重现的性能评估
    """
    
    def __init__(self, confidence_thresholds: Optional[List[float]] = None):
        """
        初始化FROC评估器
        confidence_thresholds: 置信度阈值列表，默认使用20个均匀分布的阈值
        """
        if confidence_thresholds is None:
            # 使用20个均匀分布的置信度阈值，从0.05到0.95
            self.confidence_thresholds = np.linspace(0.05, 0.95, 20).tolist()
        else:
            self.confidence_thresholds = confidence_thresholds
    
    def calculate_froc_from_predictions(self, 
                                      predictions: np.ndarray, 
                                      ground_truth: np.ndarray,
                                      confidence_scores: np.ndarray) -> Dict:
        """
        基于真实模型预测计算FROC指标
        predictions: 模型预测结果 (N, H, W, D)
        ground_truth: 真实标签 (N, H, W, D)
        confidence_scores: 置信度分数 (N, H, W, D) 
        Returns:包含FROC数据的字典
        """
        fp_rates = []
        sensitivities = []
        
        for threshold in self.confidence_thresholds:
            # 基于置信度阈值生成二值化预测
            binary_pred = (confidence_scores >= threshold).astype(int)
            
            # 计算真阳性、假阳性等
            tp = np.sum((binary_pred == 1) & (ground_truth == 1))
            fp = np.sum((binary_pred == 1) & (ground_truth == 0))
            fn = np.sum((binary_pred == 0) & (ground_truth == 1))
            
            # 计算敏感度和假阳性率
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            fp_rate = fp / np.sum(ground_truth == 0) if np.sum(ground_truth == 0) > 0 else 0
            
            sensitivities.append(sensitivity)
            fp_rates.append(fp_rate)
        
        # 计算AUC
        auc = self._calculate_auc(np.array(fp_rates), np.array(sensitivities))
        
        return {
            'fp_rates': fp_rates,
            'sensitivities': sensitivities,
            'confidence_thresholds': self.confidence_thresholds,
            'auc': auc,
            'avg_sensitivity': np.mean(sensitivities),
            'avg_fp_rate': np.mean(fp_rates),
            'data_points': len(fp_rates)
        }
    
    def _calculate_auc(self, x: np.ndarray, y: np.ndarray) -> float:
        """
        计算曲线下面积（AUC）
        x: x轴数据（假阳性率）
        y: y轴数据（敏感度）  
        Returns:AUC值
        """
        # 确保数据是排序的
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]
        
        # 使用梯形法则计算AUC
        auc = np.trapz(y_sorted, x_sorted)
        return abs(auc)
    
    def plot_froc_curve(self, 
                       froc_data: Dict, 
                       save_path: Optional[str] = None, 
                       figsize: Tuple[int, int] = (12, 8),
                       title: Optional[str] = None) -> None:
        """
        绘制FROC曲线
        froc_data: FROC数据字典
        save_path: 保存路径
        figsize: 图像大小
        title: 图表标题
        """
        plt.figure(figsize=figsize, dpi=300)
        
        fp_rates = froc_data['fp_rates']
        sensitivities = froc_data['sensitivities']
        auc = froc_data['auc']
        
        # 绘制FROC曲线
        plt.plot(fp_rates, sensitivities, 'b-', linewidth=2.5, 
                label=f'FROC曲线 (AUC = {auc:.4f})', marker='o', markersize=4)
        
        # 标注关键点
        for i in range(0, len(fp_rates), 4):  # 每4个点标注一个
            plt.annotate(f'({fp_rates[i]:.2f}, {sensitivities[i]:.2f})', 
                        (fp_rates[i], sensitivities[i]),
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=8, alpha=0.7)
        
        plt.xlabel('平均假阳性数/图像', fontsize=12)
        plt.ylabel('敏感度', fontsize=12)
        plt.title(title or 'FROC性能曲线', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=11)
        
        # 添加统计信息框
        stats_text = f"数据点数: {froc_data['data_points']}\n" + \
                    f"平均敏感度: {froc_data['avg_sensitivity']:.4f}\n" + \
                    f"平均FP率: {froc_data['avg_fp_rate']:.4f}"
        plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ FROC曲线已保存: {save_path}")
        
        plt.close()
    
    def save_froc_results(self, froc_data: Dict, save_path: str) -> None:
        """
        保存FROC结果到JSON文件
        froc_data: FROC数据字典
        save_path: 保存路径
        """
        # 确保目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(froc_data, f, indent=2, ensure_ascii=False)
        
        print(f"✓ FROC结果已保存: {save_path}")

@contextmanager
def memory_efficient_context():
    """上下文管理器"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class FROCMetric:
    """
    FROC (Free-Response Operating Characteristic) 指标计算类
    用于评估检测任务的性能，特别适用于医学图像中的病灶检测
    """
    
    def __init__(self, 
                 distance_threshold: float = 5.0,
                 confidence_thresholds: List[float] = None,
                 include_background: bool = False):
        """
        初始化FROC指标
        distance_threshold: 真阳性检测的距离阈值（像素）
        confidence_thresholds: 置信度阈值列表
        include_background: 是否包含背景类
        """
        self.distance_threshold = distance_threshold
        self.confidence_thresholds = confidence_thresholds or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.include_background = include_background
        self.reset()
    
    def reset(self):
        """重置指标状态"""
        self.all_predictions = []
        self.all_labels = []
        self.all_confidences = []
    
    def __call__(self, y_pred: torch.Tensor, y: torch.Tensor, confidence: torch.Tensor = None):
        """
        计算FROC指标
        y_pred: 预测结果 (B, C, H, W, D)
        y: 真实标签 (B, C, H, W, D)
        confidence: 置信度分数 (B, C, H, W, D)
        """
        # 转换为numpy数组
        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().cpu().numpy()
        if isinstance(y, torch.Tensor):
            y = y.detach().cpu().numpy()
        if confidence is not None and isinstance(confidence, torch.Tensor):
            confidence = confidence.detach().cpu().numpy()
        
        # 存储批次数据
        for i in range(y_pred.shape[0]):
            pred_i = y_pred[i]
            label_i = y[i]
            conf_i = confidence[i] if confidence is not None else np.ones_like(pred_i)
            
            self.all_predictions.append(pred_i)
            self.all_labels.append(label_i)
            self.all_confidences.append(conf_i)
    
    def aggregate(self) -> Dict[str, float]:
        """
        聚合所有批次的结果并计算FROC指标
        Returns:包含FROC指标的字典
        """
        if not self.all_predictions:
            return {'froc_auc': 0.0, 'sensitivity_at_fp_rates': {}}
        
        # 计算每个置信度阈值下的敏感度和假阳性率
        sensitivities = []
        fp_rates = []
        
        for threshold in self.confidence_thresholds:
            tp_total = 0
            fp_total = 0
            fn_total = 0
            total_images = len(self.all_predictions)
            
            for pred, label, conf in zip(self.all_predictions, self.all_labels, self.all_confidences):
                # 确保标签和预测的维度匹配
                if label.shape[0] == 1 and pred.shape[0] > 1:
                    # 如果标签是单通道，扩展为多通道
                    if isinstance(pred, torch.Tensor):
                        label_expanded = torch.zeros_like(pred)
                        label_expanded[0] = (label[0] == 0).float()  # 背景
                        for i in range(1, pred.shape[0]):
                            label_expanded[i] = (label[0] == i).float()  # 各类别
                    else:
                        label_expanded = np.zeros_like(pred)
                        label_expanded[0] = (label[0] == 0).astype(float)  # 背景
                        for i in range(1, pred.shape[0]):
                            label_expanded[i] = (label[0] == i).astype(float)  # 各类别
                    label = label_expanded
                
                # 对每个类别计算（跳过背景类）
                start_class = 0 if self.include_background else 1
                num_classes = min(pred.shape[0], label.shape[0])
                
                for class_idx in range(start_class, num_classes):
                    pred_class = pred[class_idx]
                    label_class = label[class_idx]
                    conf_class = conf[class_idx]
                    
                    # 应用置信度阈值
                    pred_binary = (pred_class > 0.5) & (conf_class >= threshold)
                    label_binary = label_class > 0.5
                    
                    # 计算连通组件
                    pred_components, pred_num = ndimage.label(pred_binary)
                    label_components, label_num = ndimage.label(label_binary)
                    
                    # 计算真阳性、假阳性和假阴性
                    tp, fp, fn = self._calculate_detection_metrics(
                        pred_components, pred_num, label_components, label_num
                    )
                    
                    tp_total += tp
                    fp_total += fp
                    fn_total += fn
            
            # 计算敏感度和每图像假阳性率
            sensitivity = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
            fp_rate = fp_total / total_images if total_images > 0 else 0.0
            
            sensitivities.append(sensitivity)
            fp_rates.append(fp_rate)
        
        # 计算FROC AUC
        if len(fp_rates) > 1 and len(sensitivities) > 1:
            # 确保数据是单调的
            sorted_indices = np.argsort(fp_rates)
            fp_rates_sorted = np.array(fp_rates)[sorted_indices]
            sensitivities_sorted = np.array(sensitivities)[sorted_indices]
            
            # 计算AUC
            froc_auc = auc(fp_rates_sorted, sensitivities_sorted)
        else:
            froc_auc = 0.0
        
        # 在特定假阳性率下的敏感度
        target_fp_rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        sensitivity_at_fp_rates = {}
        
        for target_fp in target_fp_rates:
            # 找到最接近目标假阳性率的敏感度
            if fp_rates:
                closest_idx = np.argmin(np.abs(np.array(fp_rates) - target_fp))
                sensitivity_at_fp_rates[f'sensitivity_at_{target_fp}_fp'] = sensitivities[closest_idx]
            else:
                sensitivity_at_fp_rates[f'sensitivity_at_{target_fp}_fp'] = 0.0
        
        return {
            'froc_auc': froc_auc,
            'mean_sensitivity': np.mean(sensitivities) if sensitivities else 0.0,
            'mean_fp_rate': np.mean(fp_rates) if fp_rates else 0.0,
            'froc_data': {
                'sensitivities': sensitivities,
                'fp_rates': fp_rates,
                'confidence_thresholds': self.confidence_thresholds
            },
            **sensitivity_at_fp_rates
        }
    
    def _calculate_detection_metrics(self, pred_components, pred_num, label_components, label_num):
        """
        计算检测指标：真阳性、假阳性、假阴性
        pred_components: 预测连通组件
        pred_num: 预测连通组件数量
        label_components: 真实连通组件
        label_num: 真实连通组件数量
        Returns:tp, fp, fn: 真阳性、假阳性、假阴性数量
        """
        tp = 0
        fp = 0
        
        # 记录已匹配的真实组件
        matched_labels = set()
        
        # 对每个预测组件，检查是否与真实组件匹配
        for pred_id in range(1, pred_num + 1):
            pred_mask = pred_components == pred_id
            pred_center = ndimage.center_of_mass(pred_mask)
            
            # 检查是否与任何真实组件匹配
            is_tp = False
            for label_id in range(1, label_num + 1):
                if label_id in matched_labels:
                    continue
                    
                label_mask = label_components == label_id
                label_center = ndimage.center_of_mass(label_mask)
                
                # 计算中心点距离
                distance = np.sqrt(sum((p - l) ** 2 for p, l in zip(pred_center, label_center)))
                
                if distance <= self.distance_threshold:
                    tp += 1
                    matched_labels.add(label_id)
                    is_tp = True
                    break
            
            if not is_tp:
                fp += 1
        
        # 假阴性 = 未匹配的真实组件
        fn = label_num - len(matched_labels)
        
        return tp, fp, fn

class ModelEvaluator:
    """
    模型评估器 
    """
    def __init__(self, 
                 model_path: str,
                 data_dir: str,
                 device: str = "cuda",
                 output_dir: str = "./evaluation_results",
                 roi_size: Tuple[int, int, int] = (96, 96, 96),
                 sw_batch_size: int = 4,
                 overlap: float = 0.5,
                 dataset_type: str = "BraTS"):
        """
        初始化评估器
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
        
        # 数据集类型和相关配置
        self.dataset_type = dataset_type
        self.num_classes = 6 if dataset_type == 'MS_MultiSpine' else 4
        
        # 推理参数
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        
        # 初始化模型类型标志
        self.is_advanced = False
        self.model_type = None
        
        # 设置随机种子
        set_determinism(seed=42)
        
        # 初始化改进的FROC评估器
        self.froc_evaluator = ImprovedFROCEvaluator()
        
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
        加载训练好的模型
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
                supported_types = ['nas_model', 'distillation_model', 'fusion_model', 'basic_model', 'nas_distillation_model']
                model_type = 'basic_model'  # 默认值
                for dir_name in parent_dirs:
                    if dir_name in supported_types:
                        model_type = dir_name
                        break
            
            # 支持的模型类型验证
            supported_types = ['nas_model', 'distillation_model', 'fusion_model', 'basic_model', 'nas_distillation_model']
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
        统一的模型加载方法
        """
        print(f"加载模型类型: {model_type}")
        
        try:
            if model_type == 'basic_model':
                # 基础模型：使用 BasicModelBank
                self.model_creator = BasicModelBank(
                    model_name=model_name,
                    device=self.device,
                    dataset_type=self.dataset_type
                )
                
                # 加载模型权重
                if 'model_state_dict' in checkpoint:
                    self.model_creator.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model_creator.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model_creator.model.load_state_dict(checkpoint)
                    
                self.model = self.model_creator.get_model()
                self.is_advanced = False
                
            elif model_type in ['nas_model', 'distillation_model', 'fusion_model', 'nas_distillation_model']:
                
                # 从检查点获取模型配置
                config = checkpoint.get('config', {})
                model_config = config.get('model_config', {})
                
                if model_type == 'nas_model':
                    # NAS模型：重建搜索出的架构
                    self.model_creator = SpecializedModelFactory(
                        model_type='nas',
                        device=self.device,
                        dataset_type=self.dataset_type,
                        **model_config
                    )
                    
                elif model_type == 'fusion_model':
                    # 融合模型：重建融合架构
                    self.model_creator = SpecializedModelFactory(
                        model_type='fusion',
                        device=self.device,
                        **model_config
                    )
                    
                elif model_type == 'distillation_model':
                    # 蒸馏学生模型：直接加载学生模型（不使用NAS架构重建）
                    # 从checkpoint中获取学生模型名称，如果没有则使用默认名称
                    student_model_name = checkpoint.get('student_model_name', model_name)
                    self.model_creator = BasicModelBank(
                        model_name=student_model_name,
                        device=self.device,
                        dataset_type=self.dataset_type
                    )
                    
                elif model_type == 'nas_distillation_model':
                    # NAS-蒸馏学生模型：重建NAS-蒸馏架构
                    self.model_creator = SpecializedModelFactory(
                        model_type='nas_distillation',
                        device=self.device,
                        dataset_type=self.dataset_type,
                        **model_config
                    )
                
                # 加载完整模型状态
                if hasattr(self.model_creator, 'model'):
                    # 对于distillation_model，尝试加载学生模型的权重
                    if model_type == 'distillation_model':
                        if 'student_state_dict' in checkpoint:
                            self.model_creator.model.load_state_dict(checkpoint['student_state_dict'])
                        elif 'model_state_dict' in checkpoint:
                            self.model_creator.model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            self.model_creator.model.load_state_dict(checkpoint['state_dict'])
                        else:
                            self.model_creator.model.load_state_dict(checkpoint)
                    else:
                        # 其他模型类型的标准加载方式
                        if 'model_state_dict' in checkpoint:
                            self.model_creator.model.load_state_dict(checkpoint['model_state_dict'])
                        elif 'state_dict' in checkpoint:
                            self.model_creator.model.load_state_dict(checkpoint['state_dict'])
                        else:
                            self.model_creator.model.load_state_dict(checkpoint)
                else:
                    # 对于复杂模型，需要特殊的加载方式
                    if 'full_model' in checkpoint:
                        self.model_creator = checkpoint['full_model']
                    else:
                        raise ValueError(f"无法加载 {model_type} 模型：缺少必要的模型信息")
                
                self.model = self.model_creator.get_model() if hasattr(self.model_creator, 'get_model') else self.model_creator
                self.is_advanced = True
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
            print(f"成功加载 {model_type} 模型: {model_name}")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            print(f"回退到基础模型加载方式...")
            
            # 回退策略：使用基础模型加载
            try:
                self.model_creator = BasicModelBank(
                    model_name=model_name,
                    device=self.device
                )
                
                if 'model_state_dict' in checkpoint:
                    self.model_creator.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    self.model_creator.model.load_state_dict(checkpoint['state_dict'])
                else:
                    self.model_creator.model.load_state_dict(checkpoint)
                    
                self.model = self.model_creator.get_model()
                self.is_advanced = False
                print(f"回退成功：以基础模型方式加载 {model_name}")
                
            except Exception as fallback_error:
                raise RuntimeError(f"无法加载模型 {model_name} (类型: {model_type}): 原始错误: {e}, 回退错误: {fallback_error}")
    

    
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
        """统一的推理接口 - 支持四种复合架构模型类型和基础模型"""
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
        print(f"设置数据加载器... (数据集类型: {self.dataset_type})")
        
        # 根据数据集类型选择相应的数据加载器
        if self.dataset_type == 'MS_MultiSpine':
            # 检查是否启用优化策略（评估时也可以使用优化的数据变换）
            use_optimization = getattr(self, 'use_optimization', True)
            
            if use_optimization:
                print("[优化策略] 评估时启用MS_MultiSpine数据集优化策略")
                # 创建优化配置（评估时不使用缓存）
                optimized_config = create_optimized_training_config(
                    data_dir=self.data_dir,
                    batch_size=1,  # 评估时使用batch_size=1
                    cache_rate=0.0,  # 评估时不使用缓存
                    num_workers=0,
                    seed=42
                )
                
                # 使用优化的数据加载器
                data_loader = optimized_config['data_loader']
                print("[优化策略] 评估时使用优化的数据变换和预处理")
            else:
                print("[标准模式] 评估时使用标准MS_MultiSpine数据加载器")
                data_loader = MSMultiSpineDatasetLoader(
                    data_dir=self.data_dir,
                    cache_rate=0.0,  # 评估时不使用缓存
                    num_workers=0,  # Windows上设置为0避免多进程问题
                    seed=42
                )
        else:  # BraTS
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
        print("启用所有评估指标")
        from monai.metrics import ConfusionMatrixMetric, MeanIoU, GeneralizedDiceScore
        
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
        
        # FROC指标
        self.froc_metric = FROCMetric(
            distance_threshold=5.0,
            include_background=False
        )
        
        # 根据数据集类型设置后处理变换
        self.post_pred = Compose([AsDiscrete(argmax=True, to_onehot=self.num_classes)])
        self.post_label = Compose([AsDiscrete(to_onehot=self.num_classes)])
        
    def evaluate_model(self) -> Dict[str, float]:
        """
        评估模型性能
        """
        print("开始模型评估...")
        
        # 重置指标
        self.dice_metric.reset()
        self.hd_metric.reset()
        self.surface_metric.reset()
        self.confusion_matrix_metric.reset()
        self.iou_metric.reset()
        self.generalized_dice_metric.reset()
        self.froc_metric.reset()
        
        all_dice_scores = []
        all_hd_scores = []
        all_surface_scores = []
        all_iou_scores = []
        all_generalized_dice_scores = []
        all_froc_scores = []
        
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
                
                # 计算所有评估指标
                dice_scores = self.dice_metric(y_pred=outputs_convert, y=labels_convert)
                hd_scores = self.hd_metric(y_pred=outputs_convert, y=labels_convert)
                surface_scores = self.surface_metric(y_pred=outputs_convert, y=labels_convert)
                iou_scores = self.iou_metric(y_pred=outputs_convert, y=labels_convert)
                generalized_dice_scores = self.generalized_dice_metric(y_pred=outputs_convert, y=labels_convert)
                confusion_matrix = self.confusion_matrix_metric(y_pred=outputs_convert, y=labels_convert)
                
                # 计算FROC指标（使用softmax输出作为置信度）
                confidence_scores = torch.softmax(outputs, dim=1)
                self.froc_metric(y_pred=outputs_convert[0], y=labels_convert[0], confidence=confidence_scores[0])
                
                # 存储单个案例结果
                case_dice = dice_scores.mean().item()
                case_hd = hd_scores.mean().item() if not torch.isnan(hd_scores.mean()) else float('inf')
                case_surface = surface_scores.mean().item() if not torch.isnan(surface_scores.mean()) else float('inf')
                case_iou = iou_scores.mean().item() if not torch.isnan(iou_scores.mean()) else 0.0
                case_generalized_dice = generalized_dice_scores.mean().item() if not torch.isnan(generalized_dice_scores.mean()) else 0.0
                
                # 处理混淆矩阵结果
                confusion_matrix_np = confusion_matrix.cpu().numpy() if hasattr(confusion_matrix, 'cpu') else confusion_matrix
                
                all_dice_scores.append(case_dice)
                all_hd_scores.append(case_hd)
                all_surface_scores.append(case_surface)
                all_iou_scores.append(case_iou)
                all_generalized_dice_scores.append(case_generalized_dice)
                
                case_result = {
                    'subject_id': subject_id,
                    'dice': case_dice,
                    'hausdorff_distance': case_hd,
                    'surface_distance': case_surface,
                    'inference_time': inference_time,
                    'iou': case_iou,
                    'generalized_dice': case_generalized_dice,
                    'confusion_matrix': confusion_matrix_np.tolist()  # 转换为列表以便JSON序列化
                }
                
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
        
        # 计算FROC指标聚合结果
        froc_results = self.froc_metric.aggregate()
        
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
            
            'total_cases': len(all_dice_scores),
            
            # FROC指标
            **froc_results
        }

        if all_iou_scores:
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
        保存详细评估结果并生成完整的可视化图表
        case_results: 每个案例的结果
        summary_results: 总体统计结果
        """
        # 保存案例级别结果
        df_cases = pd.DataFrame(case_results)
        df_cases.to_csv(self.output_dir / 'case_results.csv', index=False)
        
        # 保存总体统计
        with open(self.output_dir / 'summary_results.txt', 'w', encoding='utf-8') as f:
            f.write("医学图像分割模型评估结果\n")
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
            
            # IoU和广义Dice统计
            if 'mean_iou' in summary_results:
                f.write("IoU指标统计:\n")
                f.write(f"  平均值: {summary_results['mean_iou']:.4f} ± {summary_results['std_iou']:.4f}\n")
                f.write(f"  中位数: {summary_results['median_iou']:.4f}\n")
                f.write(f"  最小值: {summary_results['min_iou']:.4f}\n")
                f.write(f"  最大值: {summary_results['max_iou']:.4f}\n\n")
                
                f.write("广义Dice统计:\n")
                f.write(f"  平均值: {summary_results['mean_generalized_dice']:.4f} ± {summary_results['std_generalized_dice']:.4f}\n")
                f.write(f"  中位数: {summary_results['median_generalized_dice']:.4f}\n")
                f.write(f"  最小值: {summary_results['min_generalized_dice']:.4f}\n")
                f.write(f"  最大值: {summary_results['max_generalized_dice']:.4f}\n\n")
            
            # FROC指标统计
            if 'froc_auc' in summary_results:
                f.write("FROC指标统计:\n")
                f.write(f"  FROC AUC: {summary_results['froc_auc']:.4f}\n")
                f.write(f"  平均敏感度: {summary_results['mean_sensitivity']:.4f}\n")
                f.write(f"  平均假阳性率: {summary_results['mean_fp_rate']:.4f}\n")
                
                # 特定假阳性率下的敏感度
                target_fp_rates = [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
                for fp_rate in target_fp_rates:
                    key = f'sensitivity_at_{fp_rate}_fp'
                    if key in summary_results:
                        f.write(f"  敏感度@{fp_rate}FP: {summary_results[key]:.4f}\n")
                f.write("\n")
            
            f.write(f"总案例数: {summary_results['total_cases']}\n")
        
        # 生成完整的可视化图表
        self._generate_comprehensive_visualizations(case_results, summary_results)
        
        print(f"详细结果已保存到: {self.output_dir}")
    
    def _generate_comprehensive_visualizations(self, case_results: List[Dict], summary_results: Dict[str, float]):
        """
        生成所有7个指标的完整可视化图表
        """
        from utils import VisualizationUtils
        
        print("生成完整的可视化图表...")
        
        # 创建可视化目录
        viz_dir = self.output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        # 1. 基础结果分布图（原有功能）
        self._plot_results_distribution(case_results)
        
        # 2. 所有指标分布图
        try:
            VisualizationUtils.plot_all_metrics_distribution(
                case_results, 
                save_path=str(viz_dir / 'all_metrics_distribution.png')
            )
        except Exception as e:
            print(f"警告: 生成指标分布图失败: {e}")
        
        # 3. 指标对比分析
        try:
            VisualizationUtils.plot_metrics_comparison(
                case_results,
                save_path=str(viz_dir / 'metrics_comparison.png')
            )
        except Exception as e:
            print(f"警告: 生成指标对比图失败: {e}")
        
        # 4. 改进的FROC曲线（如果有FROC数据）
        if 'froc_data' in summary_results and summary_results['froc_data']:
            try:
                # 使用改进的FROC评估器生成高质量FROC曲线
                self.froc_evaluator.plot_froc_curve(
                    summary_results['froc_data'],
                    save_path=str(viz_dir / 'froc_curve.png'),
                    title=f'FROC曲线 - {self.dataset_type}数据集评估结果'
                )
                
                # 保存详细的FROC数据
                froc_json_path = viz_dir / 'froc_data.json'
                self.froc_evaluator.save_froc_results(
                    summary_results['froc_data'], 
                    str(froc_json_path)
                )
            except Exception as e:
                print(f"警告: 生成改进FROC曲线失败: {e}")
        else:
            print("警告: 没有可用的FROC数据进行可视化")
        
        # 5. 混淆矩阵热力图（如果有混淆矩阵数据）
        if hasattr(self, '_confusion_matrices') and self._confusion_matrices:
            try:
                # 根据数据集类型设置类别名称
                if self.dataset_type == 'BraTS':
                    class_names = ['坏死核心', '水肿', '增强肿瘤']
                elif self.dataset_type == 'MS_MultiSpine':
                    class_names = ['病变1', '病变2', '病变3', '病变4', '病变5']
                else:
                    class_names = [f'类别{i+1}' for i in range(self.num_classes-1)]  # 排除背景
                
                VisualizationUtils.plot_confusion_matrix_heatmap(
                    self._confusion_matrices,
                    class_names,
                    save_path=str(viz_dir / 'confusion_matrix_heatmap.png')
                )
            except Exception as e:
                print(f"警告: 生成混淆矩阵热力图失败: {e}")
        
        # 6. 指标相关性分析
        try:
            VisualizationUtils.plot_metrics_correlation(
                case_results,
                save_path=str(viz_dir / 'metrics_correlation.png')
            )
        except Exception as e:
            print(f"警告: 生成指标相关性图失败: {e}")
        
        print(f"可视化图表已保存到: {viz_dir}")
    
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
