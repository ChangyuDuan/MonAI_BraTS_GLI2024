import os
import sys
import argparse
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import json
from datetime import datetime

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

class InferenceEngine:
    """
    推理引擎 - 简化版
    专注于使用单个最佳模型进行推理
    """
    
    def __init__(self, 
                 model_path: str,
                 device: str = 'auto',
                 roi_size: Tuple[int, int, int] = (128, 128, 128),
                 sw_batch_size: int = 4,
                 overlap: float = 0.5):
        """
        初始化推理引擎
        
        Args:
            model_path: 最佳模型文件路径或训练输出目录
            device: 计算设备 ('auto', 'cpu', 'cuda')
            roi_size: 滑动窗口大小
            sw_batch_size: 滑动窗口批次大小
            overlap: 滑动窗口重叠率
        """
        self.model_path = self._resolve_model_path(model_path)
        self.device = self._setup_device(device)
        self.roi_size = roi_size
        self.sw_batch_size = sw_batch_size
        self.overlap = overlap
        
        # 加载模型
        self.model_creator = self._load_best_model()
        
        # 设置预处理和后处理
        self._setup_transforms()
        
        print(f"推理引擎初始化完成")
        print(f"设备: {self.device}")
        print(f"模型: {self.model_creator.model_name}")
        
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
        """加载最佳模型"""
        print(f"加载模型: {self.model_path}")
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # 检查是否为训练检查点
        if 'config' in checkpoint and 'model_state_dict' in checkpoint:
            # 训练检查点格式
            config = checkpoint['config']
            model_name = config.get('model_name', 'UNet')
            best_metric = checkpoint.get('best_metric', 'N/A')
            epoch = checkpoint.get('epoch', 'N/A')
            
            print(f"检测到训练检查点:")
            print(f"  模型类型: {model_name}")
            print(f"  最佳指标: {best_metric}")
            print(f"  训练轮数: {epoch}")
            
            # 创建模型
            model_creator = BasicModelBank(model_name=model_name, device=self.device)
            model_creator.model.load_state_dict(checkpoint['model_state_dict'])
            
        elif 'model_state_dict' in checkpoint:
            # 传统模型格式
            model_name = checkpoint.get('model_name', 'UNet')
            print(f"加载传统模型: {model_name}")
            
            model_creator = BasicModelBank(model_name=model_name, device=self.device)
            model_creator.model.load_state_dict(checkpoint['model_state_dict'])
            
        else:
            # 直接的状态字典
            print("加载模型状态字典")
            model_creator = BasicModelBank(model_name='UNet', device=self.device)
            model_creator.model.load_state_dict(checkpoint)
        
        model_creator.model.eval()
        return model_creator
    
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
            
            # 使用滑动窗口推理
            prediction = sliding_window_inference(
                inputs=image_tensor,
                roi_size=self.roi_size,
                sw_batch_size=self.sw_batch_size,
                predictor=self.model_creator.model,
                overlap=self.overlap,
                mode='gaussian',
                sigma_scale=0.125,
                padding_mode='constant',
                cval=0.0
            )
            
            # 应用softmax和argmax
            prediction = torch.softmax(prediction, dim=1)
            prediction = torch.argmax(prediction, dim=1, keepdim=True)
        
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
            'processing_time': datetime.now().isoformat()
        }
        
        print(f"推理完成: {output_path}")
        return result
    
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
    """
    命令行接口
    """
    parser = argparse.ArgumentParser(description='推理模块 - 简化版')
    
    # 必需参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='最佳模型文件路径或训练输出目录')
    parser.add_argument('--input', type=str, required=True,
                       help='输入文件或目录路径')
    parser.add_argument('--output', type=str, required=True,
                       help='输出文件或目录路径')
    
    # 可选参数
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='计算设备')
    parser.add_argument('--batch', action='store_true',
                       help='批量推理模式')
    parser.add_argument('--roi_size', type=int, nargs=3, default=[128, 128, 128],
                       help='滑动窗口大小')
    parser.add_argument('--sw_batch_size', type=int, default=4,
                       help='滑动窗口批次大小')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='滑动窗口重叠率')
    parser.add_argument('--no_visualization', action='store_true',
                       help='不保存可视化结果')
    
    args = parser.parse_args()
    
    try:
        # 初始化推理引擎
        engine = InferenceEngine(
            model_path=args.model_path,
            device=args.device,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap
        )
        
        # 执行推理
        save_vis = not args.no_visualization
        
        if args.batch:
            # 批量推理
            results = engine.predict_batch(
                input_dir=args.input,
                output_dir=args.output,
                save_visualization=save_vis
            )
            print(f"批量推理完成，处理了 {len(results)} 个文件")
        else:
            # 单个推理
            result = engine.predict_single_case(
                input_path=args.input,
                output_path=args.output,
                save_visualization=save_vis
            )
            print(f"推理完成: {result['output_path']}")
        
    except Exception as e:
        print(f"推理过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()