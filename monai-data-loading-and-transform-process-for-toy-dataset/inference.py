import os
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import json
from datetime import datetime
import time
import gc
from contextlib import contextmanager
import warnings

# 导入中文字体配置
try:
    from font_config import configure_chinese_font
    # 自动配置中文字体
    configure_chinese_font()
except ImportError:
    import warnings
    warnings.warn("未找到font_config模块，中文显示可能出现问题", UserWarning)

# 导入项目模块
from model import BasicModelBank
from utils import visualize_prediction_3d

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRanged, CropForegroundd, ToTensord,
    EnsureTyped, Resized, NormalizeIntensityd
)
from monai.inferers import sliding_window_inference
from monai.data import MetaTensor
from monai.utils import set_determinism

# 内存管理上下文
@contextmanager
def memory_efficient_context():
    """内存高效的上下文管理器"""
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

class InferenceEngine:
    """
    推理引擎 - 优化版本
    
    主要改进:
    - 统一的模型接口
    - 更好的内存管理
    - 增强的错误处理
    - 性能监控
    - 批量推理优化
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 roi_size: Tuple[int, int, int] = (128, 128, 128),
                 sw_batch_size: int = 4,
                 overlap: float = 0.5,
                 enable_amp: bool = True,
                 deterministic: bool = True):
        """
        初始化推理引擎
        
        Args:
            model_path: 最佳模型文件路径或训练输出目录
            device: 计算设备 ('auto', 'cpu', 'cuda')
            roi_size: 滑动窗口大小
            sw_batch_size: 滑动窗口批次大小
            overlap: 滑动窗口重叠率
            enable_amp: 是否启用自动混合精度
            deterministic: 是否使用确定性推理
        """
        # 设置确定性
        if deterministic:
            set_determinism(seed=42)
        
        self.model_path = self._resolve_model_path(model_path)
        self.device = self._setup_device(device)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        self.enable_amp = enable_amp and self.device.type == 'cuda'
        
        # 模型相关属性
        self.is_advanced = False
        self.model_type = None
        
        # 性能统计
        self.inference_stats = {
            'total_cases': 0,
            'total_time': 0.0,
            'avg_time_per_case': 0.0
        }
        
        # 加载模型
        self.model_creator = self._load_best_model()
        
        # 设置预处理和后处理
        self._setup_transforms()
        
        print(f"推理引擎初始化完成")
        print(f"设备: {self.device}")
        print(f"模型类型: {self.model_type}")
        print(f"自动混合精度: {self.enable_amp}")
        print(f"ROI大小: {self.roi_size}")
        print(f"批次大小: {self.sw_batch_size}")
        print(f"重叠率: {self.overlap}")
        
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
    
    def _setup_device(self, device: str) -> torch.device:
        """设置计算设备"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif device.lower() == 'cpu':
            return torch.device('cpu')
        elif device.lower() == 'cuda':
            if torch.cuda.is_available():
                return torch.device('cuda')
            else:
                print("警告: CUDA不可用，自动切换到CPU")
                return torch.device('cpu')
        else:
            return torch.device(device)
    
    def _load_best_model(self) -> BasicModelBank:
        """加载最佳模型 - 支持四种模型类型：nas_model、distillation_student、fusion_model、basic_model
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
            return self._load_unified_model(model_name, model_type, checkpoint)
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise RuntimeError(f"无法加载模型 {self.model_path}: {e}")
    
    def _load_unified_model(self, model_name: str, model_type: str, checkpoint: Dict):
        """
        统一的模型加载方法 - 支持保持原有架构特性
        四种模型类型：nas_model、distillation_student、fusion_model、basic_model
        """
        print(f"加载模型类型: {model_type}")
        
        try:
            if model_type == 'basic_model':
                # 基础模型：使用 BasicModelBank
                model_creator = BasicModelBank(
                    model_name=model_name,
                    device=self.device
                )
                
                # 加载模型权重
                if 'model_state_dict' in checkpoint:
                    model_creator.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model_creator.model.load_state_dict(checkpoint['state_dict'])
                else:
                    model_creator.model.load_state_dict(checkpoint)
                    
                self.is_advanced = False
                return model_creator
                
            elif model_type in ['nas_model', 'distillation_student', 'fusion_model']:
                # 高级模型：使用 SpecializedModelFactory 重建原有架构
                from model import SpecializedModelFactory
                
                # 从检查点获取模型配置
                config = checkpoint.get('config', {})
                model_config = config.get('model_config', {})
                
                if model_type == 'nas_model':
                    # NAS模型：重建搜索出的架构
                    model_creator = SpecializedModelFactory(
                        model_type='nas',
                        device=self.device,
                        **model_config
                    )
                    
                elif model_type == 'fusion_model':
                    # 融合模型：重建融合架构
                    model_creator = SpecializedModelFactory(
                        model_type='fusion',
                        device=self.device,
                        **model_config
                    )
                    
                elif model_type == 'distillation_student':
                    # 蒸馏学生模型：重建学生模型架构
                    model_creator = SpecializedModelFactory(
                        model_type='distillation',
                        device=self.device,
                        **model_config
                    )
                
                # 加载完整模型状态
                if hasattr(model_creator, 'model'):
                    if 'model_state_dict' in checkpoint:
                        model_creator.model.load_state_dict(checkpoint['model_state_dict'])
                    elif 'state_dict' in checkpoint:
                        model_creator.model.load_state_dict(checkpoint['state_dict'])
                    else:
                        model_creator.model.load_state_dict(checkpoint)
                else:
                    # 对于复杂模型，可能需要特殊的加载方式
                    if 'full_model' in checkpoint:
                        model_creator = checkpoint['full_model']
                    else:
                        raise ValueError(f"无法加载 {model_type} 模型：缺少必要的模型信息")
                
                self.is_advanced = True
                return model_creator
                
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
                
        except Exception as e:
            print(f"模型加载失败: {e}")
            print(f"回退到基础模型加载方式...")
            
            # 回退策略：使用基础模型加载
            try:
                model_creator = BasicModelBank(
                    model_name=model_name,
                    device=self.device
                )
                
                if 'model_state_dict' in checkpoint:
                    model_creator.model.load_state_dict(checkpoint['model_state_dict'])
                elif 'state_dict' in checkpoint:
                    model_creator.model.load_state_dict(checkpoint['state_dict'])
                else:
                    model_creator.model.load_state_dict(checkpoint)
                    
                self.is_advanced = False
                print(f"回退成功：以基础模型方式加载 {model_name}")
                return model_creator
                
            except Exception as fallback_error:
                raise RuntimeError(f"无法加载模型 {model_name} (类型: {model_type}): 原始错误: {e}, 回退错误: {fallback_error}")
    

    
    def _unified_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """统一的推理接口 - 支持四种模型类型"""
        with memory_efficient_context():
            if self.enable_amp:
                with torch.cuda.amp.autocast():
                    return self._perform_inference(inputs)
            else:
                return self._perform_inference(inputs)
    
    def _perform_inference(self, inputs: torch.Tensor) -> torch.Tensor:
        """执行推理"""
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
                predictor=self.model_creator.model,
                overlap=self.overlap,
                mode='gaussian',
                sigma_scale=0.125,
                padding_mode='constant',
                cval=0.0
            )
    
    def _setup_transforms(self):
        """设置预处理和后处理变换"""
        # 预处理变换 - 与训练时保持一致
        self.pre_transforms = Compose([
            LoadImaged(keys=['image']),
            EnsureChannelFirstd(keys=['image']),
            Orientationd(keys=['image'], axcodes='RAS'),
            Spacingd(keys=['image'], pixdim=(1.0, 1.0, 1.0), mode='bilinear'),
            ScaleIntensityRanged(keys=['image'], a_min=-175, a_max=250, b_min=0.0, b_max=1.0, clip=True),
            CropForegroundd(keys=['image'], source_key='image'),
            Resized(keys=['image'], spatial_size=(128, 128, 128), mode='trilinear'),
            NormalizeIntensityd(keys=['image'], nonzero=True, channel_wise=True),
            ToTensord(keys=['image'])
        ])
        
        # 后处理变换
        self.post_transforms = Compose([
            EnsureTyped(keys=['pred']),
            ToTensord(keys=['pred'])
        ])
    
    def predict_single_case(self, 
                          input_path: str, 
                          output_path: str,
                          save_visualization: bool = True) -> Dict[str, any]:
        """
        对单个病例进行推理
        
        Args:
            input_path: 输入图像路径（单个NIfTI文件或包含多模态的目录）
            output_path: 输出预测结果路径
            save_visualization: 是否保存可视化结果
            
        Returns:
            推理结果信息
        """
        print(f"开始推理: {input_path}")
        
        # 准备输入数据
        if os.path.isfile(input_path):
            # 单个文件
            data_dict = {'image': input_path}
        elif os.path.isdir(input_path):
            # 多模态目录
            modality_files = self._find_modality_files(input_path)
            if not modality_files:
                raise ValueError(f"在目录 {input_path} 中未找到有效的图像文件")
            data_dict = {'image': modality_files}
        else:
            raise ValueError(f"输入路径不存在: {input_path}")
        
        # 预处理
        data = self.pre_transforms(data_dict)
        
        # 推理
        with torch.no_grad():
            image_tensor = data['image'].unsqueeze(0).to(self.device)
            
            # 统一推理接口
            start_time = time.time()
            prediction = self._unified_inference(image_tensor)
            inference_time = time.time() - start_time
            
            # 更新性能统计
            self.inference_stats['total_cases'] += 1
            self.inference_stats['total_time'] += inference_time
            self.inference_stats['avg_time_per_case'] = (
                self.inference_stats['total_time'] / self.inference_stats['total_cases']
            )
            
            print(f"推理完成，输出形状: {prediction.shape}，耗时: {inference_time:.2f}秒")
            
            # 应用softmax和argmax（如果需要）
            if prediction.dim() > 3 and prediction.shape[1] > 1:
                # 多类别输出，应用softmax和argmax
                prediction = torch.softmax(prediction, dim=1)
                prediction = torch.argmax(prediction, dim=1, keepdim=True)
            elif prediction.dim() > 3 and prediction.shape[1] == 1:
                # 二分类输出，应用sigmoid
                prediction = torch.sigmoid(prediction)
                prediction = (prediction > 0.5).float()
        
        # 转换为numpy数组
        prediction_np = prediction.squeeze().cpu().numpy()
        
        # 保存预测结果
        self._save_prediction(prediction_np, data['image'], output_path)
        
        # 保存可视化结果
        if save_visualization:
            vis_path = output_path.replace('.nii.gz', '_visualization.png')
            original_image = data['image'].squeeze().cpu().numpy()
            if original_image.ndim == 4:  # 多模态，取第一个模态
                original_image = original_image[0]
            
            visualize_prediction_3d(
                image=original_image,
                prediction=prediction_np,
                save_path=vis_path
            )
        
        result = {
            'input_path': input_path,
            'output_path': output_path,
            'prediction_shape': prediction_np.shape,
            'unique_labels': np.unique(prediction_np).tolist(),
            'inference_time': inference_time,
            'model_type': self.model_type,
            'is_advanced': self.is_advanced,
            'processing_time': datetime.now().isoformat(),
            'performance_stats': self.inference_stats.copy()
        }
        
        print(f"推理完成: {output_path}")
        return result
    
    def get_performance_report(self) -> Dict[str, any]:
        """获取性能报告"""
        return {
            'inference_stats': self.inference_stats.copy(),
            'model_info': {
                'type': self.model_type,
                'is_advanced': self.is_advanced,
                'device': str(self.device),
                'enable_amp': self.enable_amp
            },
            'config': {
                'roi_size': self.roi_size,
                'sw_batch_size': self.sw_batch_size,
                'overlap': self.overlap
            }
        }
    
    def print_performance_summary(self):
        """打印性能摘要"""
        stats = self.inference_stats
        print("\n=== 推理性能摘要 ===")
        print(f"总处理案例数: {stats['total_cases']}")
        print(f"总推理时间: {stats['total_time']:.2f} 秒")
        print(f"平均每案例时间: {stats['avg_time_per_case']:.2f} 秒")
        print(f"模型类型: {self.model_type} ({'高级' if self.is_advanced else '基础'})")
        print(f"设备: {self.device}")
        print(f"自动混合精度: {self.enable_amp}")
        print("===================")
    
    def predict_batch(self, 
                     input_dir: str, 
                     output_dir: str,
                     save_visualization: bool = True) -> List[Dict[str, any]]:
        """
        批量推理
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            save_visualization: 是否保存可视化结果
            
        Returns:
            批量推理结果列表
        """
        print(f"开始批量推理: {input_dir} -> {output_dir}")
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 查找输入文件
        input_files = self._find_input_files(input_dir)
        if not input_files:
            raise ValueError(f"在目录 {input_dir} 中未找到有效的输入文件")
        
        results = []
        
        # 批量处理
        for input_file in tqdm(input_files, desc="批量推理"):
            try:
                # 生成输出文件名
                input_name = Path(input_file).stem
                if input_name.endswith('.nii'):
                    input_name = input_name[:-4]
                output_file = os.path.join(output_dir, f"{input_name}_prediction.nii.gz")
                
                # 单个推理
                result = self.predict_single_case(
                    input_path=input_file,
                    output_path=output_file,
                    save_visualization=save_visualization
                )
                results.append(result)
                
                # 清理内存缓存，避免内存累积
                if torch.cuda.is_available() and self.device.type == 'cuda':
                    torch.cuda.empty_cache()
                # 强制垃圾回收，清理CPU内存
                import gc
                gc.collect()
                
            except Exception as e:
                print(f"处理文件 {input_file} 时出错: {e}")
                results.append({
                    'input_path': input_file,
                    'error': str(e),
                    'processing_time': datetime.now().isoformat()
                })
        
        # 保存批量处理报告
        report_path = os.path.join(output_dir, 'batch_inference_report.json')
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"批量推理完成，共处理 {len(input_files)} 个文件")
        print(f"结果报告保存到: {report_path}")
        
        return results
    
    def _find_modality_files(self, directory: str) -> List[str]:
        """在目录中查找多模态文件"""
        modalities = ['t1', 't1ce', 't2', 'flair']
        files = []
        
        for modality in modalities:
            pattern_files = list(Path(directory).glob(f"*{modality}*.nii.gz"))
            if pattern_files:
                files.append(str(pattern_files[0]))
        
        return files if len(files) > 0 else []
    
    def _find_input_files(self, directory: str) -> List[str]:
        """查找输入文件"""
        extensions = ['*.nii.gz', '*.nii']
        files = []
        
        for ext in extensions:
            files.extend(list(Path(directory).glob(ext)))
        
        return [str(f) for f in files]
    
    def _save_prediction(self, prediction: np.ndarray, reference_image: MetaTensor, output_path: str):
        """保存预测结果"""
        # 创建NIfTI图像
        if hasattr(reference_image, 'affine'):
            affine = reference_image.affine.numpy()
        else:
            affine = np.eye(4)
        
        # 创建NIfTI对象
        nifti_img = nib.Nifti1Image(prediction.astype(np.uint8), affine)
        
        # 保存文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        nib.save(nifti_img, output_path)

def main():
    pass

if __name__ == '__main__':
    main()

# 注意: 此文件主要通过 main.py 调用，不提供独立的命令行接口
# 如需独立使用推理功能，请使用: python main.py --mode inference