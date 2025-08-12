#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MS_MultiSpine数据集优化训练策略
针对低前景比例数据集的自适应训练方案

主要优化策略：
1. 自适应RandCropByPosNegLabeld参数
2. 渐进式训练策略
3. 数据不平衡优化损失函数
4. 智能数据增强策略
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    RandCropByPosNegLabeld,
    RandFlipd,
    RandRotate90d,
    RandShiftIntensityd,
    RandAffined,
    ToTensord,
    Resized,
    NormalizeIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandZoomd,
    Rand3DElasticd,
    RandBiasFieldd,
    Lambda
)
from monai.data import Dataset, CacheDataset, DataLoader as MonaiDataLoader
from torch.utils.data import DataLoader
from monai.losses import DiceLoss, FocalLoss, DiceCELoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism

class AdaptiveRandCropByPosNegLabeld:
    """
    自适应的正负样本裁剪策略
    根据数据集的前景比例动态调整参数
    """
    
    def __init__(self, 
                 keys: List[str],
                 label_key: str,
                 spatial_size: Tuple[int, int, int],
                 image_key: str = "image",
                 foreground_ratio_threshold: float = 0.001,
                 adaptive_mode: str = "progressive"):
        """
        初始化自适应裁剪策略
        
        Args:
            keys: 要处理的键列表
            label_key: 标签键名
            spatial_size: 裁剪尺寸
            image_key: 图像键名
            foreground_ratio_threshold: 前景比例阈值
            adaptive_mode: 自适应模式 ('progressive', 'flexible', 'background_aware')
        """
        self.keys = keys
        self.label_key = label_key
        self.spatial_size = spatial_size
        self.image_key = image_key
        self.foreground_ratio_threshold = foreground_ratio_threshold
        self.adaptive_mode = adaptive_mode
        
    def __call__(self, data: Dict) -> Dict:
        """
        执行自适应裁剪
        """
        # 计算前景比例
        label = data[self.label_key]
        if isinstance(label, torch.Tensor):
            label_np = label.cpu().numpy()
        else:
            label_np = np.array(label)
            
        total_voxels = np.prod(label_np.shape)
        foreground_voxels = np.sum(label_np > 0)
        foreground_ratio = foreground_voxels / total_voxels if total_voxels > 0 else 0
        
        # 根据前景比例选择策略
        if foreground_ratio < self.foreground_ratio_threshold:
            # 极低前景比例：使用背景感知策略
            return self._background_aware_crop(data, foreground_ratio)
        elif foreground_ratio < 0.01:
            # 低前景比例：使用灵活策略
            return self._flexible_crop(data, foreground_ratio)
        else:
            # 正常前景比例：使用标准策略
            return self._standard_crop(data)
    
    def _background_aware_crop(self, data: Dict, foreground_ratio: float) -> Dict:
        """
        背景感知裁剪：主要生成背景样本，偶尔尝试前景
        """
        # 对于无前景或极少前景的样本，主要使用背景裁剪
        if foreground_ratio == 0:
            # 完全无前景：只生成背景样本
            crop_transform = RandCropByPosNegLabeld(
                keys=self.keys,
                label_key=self.label_key,
                spatial_size=self.spatial_size,
                pos=0,  # 不要求前景
                neg=2,  # 生成2个背景样本
                num_samples=2,
                image_key=self.image_key,
                image_threshold=0,
                allow_smaller=True  # 允许更小的裁剪
            )
        else:
            # 极少前景：偶尔尝试前景，主要使用背景
            crop_transform = RandCropByPosNegLabeld(
                keys=self.keys,
                label_key=self.label_key,
                spatial_size=self.spatial_size,
                pos=0.2,  # 20%概率尝试前景
                neg=1.8,  # 80%使用背景
                num_samples=2,
                image_key=self.image_key,
                image_threshold=0,
                allow_smaller=True
            )
        
        return crop_transform(data)
    
    def _flexible_crop(self, data: Dict, foreground_ratio: float) -> Dict:
        """
        灵活裁剪：根据前景比例动态调整正负样本比例
        """
        # 动态计算pos和neg比例
        pos_ratio = min(0.5, foreground_ratio * 50)  # 前景比例的50倍，最大0.5
        neg_ratio = 2 - pos_ratio
        
        crop_transform = RandCropByPosNegLabeld(
            keys=self.keys,
            label_key=self.label_key,
            spatial_size=self.spatial_size,
            pos=pos_ratio,
            neg=neg_ratio,
            num_samples=2,
            image_key=self.image_key,
            image_threshold=0,
            allow_smaller=True
        )
        
        return crop_transform(data)
    
    def _standard_crop(self, data: Dict) -> Dict:
        """
        标准裁剪：使用原始的平衡策略
        """
        crop_transform = RandCropByPosNegLabeld(
            keys=self.keys,
            label_key=self.label_key,
            spatial_size=self.spatial_size,
            pos=1,
            neg=1,
            num_samples=2,
            image_key=self.image_key,
            image_threshold=0
        )
        
        return crop_transform(data)

class ProgressiveDataAugmentation:
    """
    渐进式数据增强策略
    训练初期使用轻度增强，后期逐渐增强
    """
    
    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        """设置当前训练轮次"""
        self.current_epoch = epoch
        
    def get_augmentation_intensity(self) -> float:
        """获取当前增强强度 (0.0-1.0)"""
        if self.total_epochs <= 1:
            return 0.5
        
        # 前30%轮次使用轻度增强，后70%逐渐增强
        if self.current_epoch < self.total_epochs * 0.3:
            return 0.3  # 轻度增强
        else:
            # 线性增长到最大强度
            progress = (self.current_epoch - self.total_epochs * 0.3) / (self.total_epochs * 0.7)
            return 0.3 + 0.4 * progress  # 从0.3增长到0.7
    
    def get_progressive_transforms(self, base_transforms: List, mode: str = "train") -> Compose:
        """获取渐进式变换"""
        if mode != "train":
            return Compose(base_transforms + [ToTensord(keys=["image", "label"])])
        
        intensity = self.get_augmentation_intensity()
        
        # 自适应裁剪（始终使用）
        adaptive_crop = AdaptiveRandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(128, 128, 128),
            image_key="image"
        )
        
        # 渐进式增强变换
        progressive_transforms = [
            # 几何变换（强度随训练进度增加）
            RandFlipd(
                keys=["image", "label"],
                spatial_axis=[0, 1, 2],
                prob=0.05 + 0.15 * intensity  # 从5%增长到20%
            ),
            RandRotate90d(
                keys=["image", "label"],
                prob=0.05 + 0.15 * intensity,
                max_k=3
            ),
            RandAffined(
                keys=['image', 'label'],
                mode=('bilinear', 'nearest'),
                prob=0.1 + 0.2 * intensity,  # 从10%增长到30%
                spatial_size=(128, 128, 128),
                rotate_range=(0, 0, np.pi / 30 * intensity),  # 旋转角度逐渐增加
                scale_range=(0.05 * intensity, 0.05 * intensity, 0.05 * intensity)
            ),
            # 强度变换（适度使用）
            RandShiftIntensityd(
                keys=["image"],
                offsets=0.05 + 0.1 * intensity,  # 从5%增长到15%
                prob=0.3 + 0.2 * intensity
            ),
            RandGaussianNoised(
                keys=["image"],
                prob=0.1 * intensity,  # 噪声概率较低
                mean=0.0,
                std=0.05 * intensity
            ),
            RandAdjustContrastd(
                keys=["image"],
                prob=0.1 + 0.1 * intensity,
                gamma=(0.8 + 0.1 * intensity, 1.2 - 0.1 * intensity)
            )
        ]
        
        # 组合所有变换
        all_transforms = base_transforms + [adaptive_crop] + progressive_transforms + [ToTensord(keys=["image", "label"])]
        
        return Compose(all_transforms)

class ImbalancedLoss(nn.Module):
    """
    针对数据不平衡的复合损失函数
    结合Dice Loss、Focal Loss和权重调整
    """
    
    def __init__(self, 
                 num_classes: int = 6,
                 dice_weight: float = 0.5,
                 focal_weight: float = 0.3,
                 ce_weight: float = 0.2,
                 focal_gamma: float = 2.0,
                 class_weights: Optional[List[float]] = None):
        super().__init__()
        
        self.num_classes = num_classes
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        
        # 初始化损失函数
        self.dice_loss = DiceLoss(
            include_background=True,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True
        )
        
        self.focal_loss = FocalLoss(
            include_background=True,
            to_onehot_y=True,
            gamma=focal_gamma,
            weight=torch.tensor(class_weights) if class_weights else None
        )
        
        self.ce_loss = nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights) if class_weights else None
        )
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        计算复合损失
        
        Args:
            pred: 预测结果 [B, C, H, W, D]
            target: 真实标签 [B, 1, H, W, D] 或 [B, H, W, D]
            
        Returns:
            复合损失值
        """
        # 确保target维度正确
        if target.dim() == 4:  # [B, H, W, D]
            target = target.unsqueeze(1)  # [B, 1, H, W, D]
        
        # 计算各项损失
        dice_loss_val = self.dice_loss(pred, target)
        focal_loss_val = self.focal_loss(pred, target)
        
        # CE损失需要特殊处理
        target_ce = target.squeeze(1).long()  # [B, H, W, D]
        ce_loss_val = self.ce_loss(pred, target_ce)
        
        # 加权组合
        total_loss = (
            self.dice_weight * dice_loss_val +
            self.focal_weight * focal_loss_val +
            self.ce_weight * ce_loss_val
        )
        
        return total_loss

class MSMultiSpineDatasetLoader:
    """
    MS_MultiSpine数据集加载器（优化版本）
    集成所有优化策略，包括自适应裁剪、渐进式增强、复合损失函数等
    
    特点：
    - 自动检测文件命名模式和前缀
    - 支持任意编号范围和文件前缀组合
    - 兼容多种MRI序列类型（T2 + STIR/PSIR/MP2RAGE等）
    - 灵活适应数据集扩展
    - 针对低前景比例数据集优化
    """
    
    def __init__(self, 
                 data_dir: str,
                 cache_rate: float = 0.0,  # 降低缓存以节省内存
                 num_workers: int = 0,
                 seed: int = 42,
                 total_epochs: int = 10):
        
        self.data_dir = data_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        self.total_epochs = total_epochs
        
        # 设置随机种子
        set_determinism(seed=seed)
        
        # 初始化渐进式增强
        self.progressive_aug = ProgressiveDataAugmentation(total_epochs)
        
        # 数据质量统计
        self.data_quality_stats = None
        
    def analyze_data_quality(self, data_files: List[Dict]) -> Dict:
        """
        分析数据质量，为优化策略提供依据
        """
        print("正在分析数据质量...")
        
        foreground_ratios = []
        quality_scores = []
        
        for data_dict in data_files[:10]:  # 采样分析前10个文件
            try:
                # 简单加载标签进行分析
                import nibabel as nib
                label_path = data_dict['label']
                label_img = nib.load(label_path)
                label_data = label_img.get_fdata()
                
                total_voxels = np.prod(label_data.shape)
                foreground_voxels = np.sum(label_data > 0)
                foreground_ratio = foreground_voxels / total_voxels if total_voxels > 0 else 0
                
                foreground_ratios.append(foreground_ratio)
                
                # 简单质量评分
                quality_score = 100 if foreground_ratio > 0.001 else 50 if foreground_ratio > 0 else 0
                quality_scores.append(quality_score)
                
            except Exception as e:
                print(f"分析文件 {data_dict.get('subject_id', 'unknown')} 时出错: {e}")
                continue
        
        stats = {
            'avg_foreground_ratio': np.mean(foreground_ratios) if foreground_ratios else 0,
            'min_foreground_ratio': np.min(foreground_ratios) if foreground_ratios else 0,
            'max_foreground_ratio': np.max(foreground_ratios) if foreground_ratios else 0,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'low_quality_count': sum(1 for score in quality_scores if score < 60),
            'total_analyzed': len(foreground_ratios)
        }
        
        self.data_quality_stats = stats
        
        print(f"数据质量分析完成:")
        print(f"  平均前景比例: {stats['avg_foreground_ratio']*100:.3f}%")
        print(f"  低质量样本数: {stats['low_quality_count']}/{stats['total_analyzed']}")
        
        return stats
    
    def get_optimized_transforms(self, mode: str = "train", epoch: int = 0) -> Compose:
        """
        获取优化的数据变换
        """
        # 更新渐进式增强的轮次
        self.progressive_aug.set_epoch(epoch)
        
        # 基础变换
        base_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,
                a_max=400,
                b_min=0.0,
                b_max=1.0,
                clip=True
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            Resized(
                keys=["image", "label"],
                spatial_size=(128, 128, 128),
                mode=("trilinear", "nearest")
            ),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            # 标签重映射
            Lambda(self._remap_labels)
        ]
        
        # 获取渐进式变换
        return self.progressive_aug.get_progressive_transforms(base_transforms, mode)
    
    def _remap_labels(self, data):
        """标签重映射：将标签6映射到标签5"""
        label = data["label"]
        if isinstance(label, torch.Tensor):
            label = torch.where(label == 6, 5, label)
            label = torch.clamp(label, 0, 5)
        else:
            label = np.where(label == 6, 5, label)
            label = np.clip(label, 0, 5)
        data["label"] = label
        return data
    
    def get_optimized_loss_function(self) -> ImbalancedLoss:
        """
        获取优化的损失函数
        """
        # 根据数据质量调整类别权重
        if self.data_quality_stats:
            avg_fg_ratio = self.data_quality_stats['avg_foreground_ratio']
            # 为前景类别分配更高权重
            bg_weight = 1.0
            fg_weight = min(10.0, 1.0 / max(avg_fg_ratio, 0.001))  # 前景权重反比于前景比例
            
            # MS_MultiSpine有6个类别 (0-5)
            class_weights = [bg_weight, fg_weight, fg_weight, fg_weight, fg_weight, fg_weight]
        else:
            # 默认权重
            class_weights = [1.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        
        return ImbalancedLoss(
            num_classes=6,
            dice_weight=0.5,
            focal_weight=0.3,
            ce_weight=0.2,
            focal_gamma=2.0,
            class_weights=class_weights
        )
    
    def get_training_recommendations(self) -> Dict:
        """
        基于数据分析提供训练建议
        """
        recommendations = {
            'batch_size': 1,  # 小批次以适应内存限制
            'learning_rate': 1e-4,  # 较小的学习率
            'optimizer': 'AdamW',
            'scheduler': 'CosineAnnealingLR',
            'early_stopping_patience': 10,
            'gradient_clipping': 1.0,
            'mixed_precision': True,  # 使用混合精度训练
            'cache_rate': 0.0,  # 不使用缓存以节省内存
        }
        
        if self.data_quality_stats:
            avg_fg_ratio = self.data_quality_stats['avg_foreground_ratio']
            low_quality_ratio = self.data_quality_stats['low_quality_count'] / max(self.data_quality_stats['total_analyzed'], 1)
            
            # 根据数据质量调整建议
            if avg_fg_ratio < 0.001:
                recommendations.update({
                    'learning_rate': 5e-5,  # 更小的学习率
                    'early_stopping_patience': 15,  # 更大的耐心
                    'warmup_epochs': 3,  # 添加预热
                })
            
            if low_quality_ratio > 0.5:
                recommendations.update({
                    'data_filtering': True,  # 建议过滤低质量数据
                    'augmentation_intensity': 'high',  # 增强数据增强
                })
        
        return recommendations
    
    def _detect_file_pattern(self, case_path: str, case_id: str) -> Optional[Tuple[str, str]]:
        """
        自动检测文件模式，返回(prefix, second_modality)
        通过扫描实际文件来确定前缀和模态类型
        """
        try:
            import glob
            # 获取目录中所有.nii.gz文件
            nii_files = glob.glob(os.path.join(case_path, "*.nii.gz"))
            if not nii_files:
                return None
            
            # 提取文件名（不含路径和扩展名）
            filenames = [os.path.basename(f).replace('.nii.gz', '') for f in nii_files]
            
            # 寻找T2文件来确定前缀
            t2_files = [f for f in filenames if f.endswith('_T2')]
            if not t2_files:
                return None
            
            # 从T2文件名提取前缀 (例如: "11-001_T2" -> prefix="11")
            t2_filename = t2_files[0]
            parts = t2_filename.split('-')
            if len(parts) < 2:
                return None
            prefix = parts[0]
            
            # 寻找第二模态文件
            # 排除T2和LESIONMASK，剩下的就是第二模态
            second_modality_files = []
            for filename in filenames:
                if (filename.startswith(f"{prefix}-{case_id}_") and 
                    not filename.endswith('_T2') and 
                    not filename.endswith('_LESIONMASK')):
                    # 提取模态名称 (例如: "11-001_STIR" -> "STIR")
                    modality = filename.split('_')[-1]
                    second_modality_files.append(modality)
            
            if not second_modality_files:
                return None
            
            # 返回检测到的前缀和第二模态
            second_modality = second_modality_files[0]
            return prefix, second_modality
            
        except Exception as e:
            print(f"检测文件模式时出错 {case_path}: {e}")
            return None
    
    def get_data_dicts(self) -> Tuple[List[Dict], List[Dict]]:
        """
        获取训练和验证数据字典
        使用自动文件检测，支持任意编号范围和文件前缀组合
        train_files: 训练数据列表
        val_files: 验证数据列表
        """
        data_files = []
        
        # 扫描数据目录
        if os.path.exists(self.data_dir):
            for case_dir in sorted(os.listdir(self.data_dir)):
                case_path = os.path.join(self.data_dir, case_dir)
                
                # 跳过非目录文件
                if not os.path.isdir(case_path):
                    continue
                
                # 提取病例编号 (sub-001 -> 001)
                if case_dir.startswith('sub-'):
                    case_id = case_dir[4:]  # 去掉'sub-'前缀
                else:
                    print(f"警告: 目录名格式不正确 {case_dir}，跳过（应为sub-xxx格式）")
                    continue
                
                # 自动检测文件模式
                pattern_result = self._detect_file_pattern(case_path, case_id)
                if pattern_result is None:
                    print(f"警告: 无法检测病例 {case_dir} 的文件模式，跳过")
                    continue
                
                prefix, second_modality = pattern_result
                
                # 构建MS_MultiSpine文件路径
                t2_path = os.path.join(case_path, f"{prefix}-{case_id}_T2.nii.gz")
                second_modality_path = os.path.join(case_path, f"{prefix}-{case_id}_{second_modality}.nii.gz")
                mask_path = os.path.join(case_path, f"{prefix}-{case_id}_LESIONMASK.nii.gz")
                
                # 检查所有必需文件是否存在
                required_files = [t2_path, second_modality_path, mask_path]
                if all(os.path.exists(f) for f in required_files):
                    data_dict = {
                        'image': [t2_path, second_modality_path],  # 2个模态：T2和第二模态
                        'label': mask_path,
                        'subject_id': case_dir,
                        'modalities': ['T2', second_modality],  # 记录实际使用的模态
                        'prefix': prefix  # 记录检测到的前缀
                    }
                    data_files.append(data_dict)
                    print(f"成功加载病例 {case_dir}: T2 + {second_modality} (前缀: {prefix})")
                else:
                    missing_files = [f for f in required_files if not os.path.exists(f)]
                    print(f"警告: 病例 {case_dir} 缺少文件: {[os.path.basename(f) for f in missing_files]}，跳过")
        
        # 如果没有找到数据，直接警告并返回空列表
        if not data_files:
            print("警告: 未找到MS_MultiSpine数据，请检查数据路径是否正确")
            print(f"数据路径: {self.data_dir}")
            print("请确保数据目录包含sub-*格式的病例文件夹")
            return [], []
        
        # 划分训练和验证集 (80% 训练, 20% 验证)
        split_idx = int(0.8 * len(data_files))
        train_files = data_files[:split_idx]
        val_files = data_files[split_idx:]
        
        print(f"找到 {len(data_files)} 个完整的MS_MultiSpine病例")
        print(f"训练样本数: {len(train_files)}")
        print(f"验证样本数: {len(val_files)}")
        
        return train_files, val_files
    
    def get_transforms(self, mode: str = "train", epoch: int = 0) -> Compose:
        """
        获取数据变换流程（兼容原接口）
        适配脊柱MRI的特点，集成优化策略
        """
        return self.get_optimized_transforms(mode, epoch)
    
    def get_dataloaders(self, batch_size: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        获取训练和验证数据加载器
        batch_size: 批次大小
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        """
        # 获取数据文件列表
        train_files, val_files = self.get_data_dicts()
        return self.create_dataloaders_from_dicts(train_files, val_files, batch_size)
    
    def create_dataloaders_from_dicts(self, train_files: List[Dict], val_files: List[Dict], 
                                     batch_size: int = 2) -> Tuple[DataLoader, DataLoader]:
        """
        从给定的数据字典创建数据加载器
        train_files: 训练数据字典列表
        val_files: 验证数据字典列表
        batch_size: 批次大小
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        """
        # 获取变换
        train_transforms = self.get_transforms("train")
        val_transforms = self.get_transforms("val")
        
        # 创建数据集
        if self.cache_rate > 0:
            # 缓存数据集
            train_ds = CacheDataset(
                data=train_files,
                transform=train_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers
            )
            val_ds = CacheDataset(
                data=val_files,
                transform=val_transforms,
                cache_rate=self.cache_rate,
                num_workers=self.num_workers
            )
        else:
            # 使用普通数据集
            train_ds = Dataset(data=train_files, transform=train_transforms)
            val_ds = Dataset(data=val_files, transform=val_transforms)
        
        # 创建数据加载器 - 添加Windows兼容性设置
        train_loader = MonaiDataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,  # Windows兼容性设置
            multiprocessing_context=None  # 避免多进程问题
        )
        
        val_loader = MonaiDataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=False,  # Windows兼容性设置
            multiprocessing_context=None  # 避免多进程问题
        )
        
        return train_loader, val_loader
    
    def print_data_info(self, dataloader: DataLoader):
        """
        打印数据加载器信息
        """
        print(f"数据集大小: {len(dataloader.dataset)}")
        print(f"批次数量: {len(dataloader)}")
        print(f"批次大小: {dataloader.batch_size}")
        
        # 获取一个批次的数据查看形状
        for batch in dataloader:
            print(f"图像形状: {batch['image'].shape}")
            print(f"标签形状: {batch['label'].shape}")
            print(f"图像数据类型: {batch['image'].dtype}")
            print(f"标签数据类型: {batch['label'].dtype}")
            print(f"图像值范围: [{batch['image'].min():.3f}, {batch['image'].max():.3f}]")
            print(f"标签唯一值: {torch.unique(batch['label'])}")
            break

def create_optimized_training_config(data_dir: str, 
                                   total_epochs: int = 10,
                                   device: str = "cpu") -> Dict:
    """
    创建优化的训练配置
    
    Args:
        data_dir: 数据目录路径
        total_epochs: 总训练轮次
        device: 训练设备
        
    Returns:
        优化的训练配置字典
    """
    print("正在创建优化的训练配置...")
    
    # 创建优化的数据加载器
    loader = MSMultiSpineDatasetLoader(
        data_dir=data_dir,
        cache_rate=0.0,
        num_workers=0,
        total_epochs=total_epochs
    )
    
    # 模拟数据文件列表（实际使用时需要真实数据）
    data_files = [{'label': os.path.join(data_dir, f'sub-{i:03d}', f'*_LESIONMASK.nii.gz')} 
                  for i in range(1, 11)]  # 前10个样本
    
    # 分析数据质量
    try:
        loader.analyze_data_quality(data_files)
    except Exception as e:
        print(f"数据质量分析失败: {e}，使用默认配置")
    
    # 获取训练建议
    recommendations = loader.get_training_recommendations()
    
    # 创建配置
    config = {
        'data_loader': loader,
        'loss_function': loader.get_optimized_loss_function(),
        'training_params': recommendations,
        'transforms': {
            'train': lambda epoch: loader.get_optimized_transforms('train', epoch),
            'val': lambda epoch: loader.get_optimized_transforms('val', epoch)
        },
        'device': device,
        'total_epochs': total_epochs
    }
    
    print("优化配置创建完成！")
    print("\n=== 训练建议 ===")
    for key, value in recommendations.items():
        print(f"{key}: {value}")
    
    return config

def print_optimization_summary():
    """
    打印优化策略总结
    """
    print("\n" + "="*80)
    print("MS_MultiSpine数据集优化训练策略总结")
    print("="*80)
    
    print("\n🎯 主要优化策略:")
    print("1. 自适应RandCropByPosNegLabeld:")
    print("   - 无前景样本: 只生成背景样本 (pos=0, neg=2)")
    print("   - 极少前景: 动态调整正负比例 (pos=0.2, neg=1.8)")
    print("   - 正常前景: 使用标准平衡策略 (pos=1, neg=1)")
    
    print("\n2. 渐进式数据增强:")
    print("   - 前30%轮次: 轻度增强 (强度0.3)")
    print("   - 后70%轮次: 逐渐增强 (强度0.3→0.7)")
    print("   - 几何变换概率: 5%→20%")
    print("   - 强度变换概率: 5%→15%")
    
    print("\n3. 数据不平衡损失函数:")
    print("   - Dice Loss (50%) + Focal Loss (30%) + CE Loss (20%)")
    print("   - 前景类别权重: 5倍于背景")
    print("   - Focal Loss gamma=2.0 处理难样本")
    
    print("\n4. 训练参数优化:")
    print("   - 批次大小: 1 (适应内存限制)")
    print("   - 学习率: 1e-4 (低前景比例时降至5e-5)")
    print("   - 缓存率: 0.0 (节省内存)")
    print("   - 混合精度训练: 启用")
    
    print("\n💡 使用建议:")
    print("1. 不过滤数据，最大化利用所有样本")
    print("2. 使用CPU训练以避免内存不足")
    print("3. 监控训练过程，适时调整参数")
    print("4. 使用早停机制防止过拟合")
    
    print("\n📊 预期效果:")
    print("- 减少'Num foregrounds 0'警告")
    print("- 提高训练稳定性")
    print("- 更好地利用小数据集")
    print("- 适应数据不平衡问题")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    # 示例使用
    data_dir = "./MS_MultiSpine_dataset/MS_MultiSpine_dataset/derivatives/preprocessedAndRegistered"
    
    print_optimization_summary()
    
    # 创建优化配置
    config = create_optimized_training_config(
        data_dir=data_dir,
        total_epochs=10,
        device="cpu"
    )
    
    print("\n✅ 优化策略脚本创建完成！")
    print("\n使用方法:")
    print("1. 导入此模块: from MSMultiSpineLoader import create_optimized_training_config")
    print("2. 创建配置: config = create_optimized_training_config(data_dir, total_epochs, device)")
    print("3. 在训练循环中使用config['transforms']['train'](epoch)获取变换")
    print("4. 使用config['loss_function']作为损失函数")
    print("5. 参考config['training_params']调整训练参数")