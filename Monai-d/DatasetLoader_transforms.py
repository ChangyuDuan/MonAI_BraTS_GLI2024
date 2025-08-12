import os
import glob
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from monai.data import Dataset, CacheDataset, DataLoader as MonaiDataLoader
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
    EnsureTyped,
    Resized,
    NormalizeIntensityd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandAdjustContrastd,
    RandZoomd,
    Rand3DElasticd,
    RandBiasFieldd,
    MapLabelValued  
)
from monai.utils import set_determinism

class DatasetLoader:
    """
    数据集加载器
    支持多模态MRI图像（T1, T1ce, T2, FLAIR）和分割标签
    """
    
    def __init__(self, 
                 data_dir: str,
                 cache_rate: float = 1.0,
                 num_workers: int = 0,  # Windows上默认设置为0避免多进程问题
                 seed: int = 42):
        """
        初始化数据加载器
        data_dir: 数据集根目录
        cache_rate: 缓存比例 (0.0-1.0)
        num_workers: 数据加载工作进程数
        seed: 随机种子
        """
        self.data_dir = data_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        
        # 设置随机种子
        set_determinism(seed=seed)
        
        # 定义图像模态
        self.modalities = ['t1n', 't1c', 't2f', 't2w']
        
    def get_data_dicts(self) -> Tuple[List[Dict], List[Dict]]:
        """
        获取训练和验证数据字典
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
                
                # 构建文件路径
                t1n_path = os.path.join(case_path, f"{case_dir}-t1n.nii.gz")
                t1c_path = os.path.join(case_path, f"{case_dir}-t1c.nii.gz")
                t2w_path = os.path.join(case_path, f"{case_dir}-t2w.nii.gz")
                t2f_path = os.path.join(case_path, f"{case_dir}-t2f.nii.gz")
                seg_path = os.path.join(case_path, f"{case_dir}-seg.nii.gz")
                
                # 检查所有必需文件是否存在
                required_files = [t1n_path, t1c_path, t2w_path, t2f_path, seg_path]
                if all(os.path.exists(f) for f in required_files):
                    data_dict = {
                        'image': [t1n_path, t1c_path, t2w_path, t2f_path],
                        'label': seg_path,
                        'subject_id': case_dir
                    }
                    data_files.append(data_dict)
                else:
                    missing_files = [f for f in required_files if not os.path.exists(f)]
                    print(f"警告: 病例 {case_dir} 缺少文件: {[os.path.basename(f) for f in missing_files]}，跳过")
        
        # 如果没有真实数据，直接警告并返回空列表
        if not data_files:
            print("警告: 未找到真实BraTS数据，请检查数据路径是否正确")
            print(f"数据路径: {self.data_dir}")
            print("请确保数据目录包含BraTS-GLI-*格式的病例文件夹")
            return [], []
        
        # 划分训练和验证集 (80% 训练, 20% 验证)
        split_idx = int(0.8 * len(data_files))
        train_files = data_files[:split_idx]
        val_files = data_files[split_idx:]
        
        print(f"找到 {len(data_files)} 个完整的病例")
        print(f"训练样本数: {len(train_files)}")
        print(f"测试样本数: {len(val_files)}")
        
        return train_files, val_files
    def get_transforms(self, mode: str = "train") -> Compose:
        """
        获取数据变换流程
        """
        # 基础变换（训练和验证都使用）
        base_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # 添加标签重映射：将BraTS标签值4映射为3
            MapLabelValued(keys=["label"], orig_labels=[0, 1, 2, 4], target_labels=[0, 1, 2, 3]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
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
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True)
        ]
        
        if mode == "train":
            # 训练时添加数据增强
            train_transforms = base_transforms + [
                RandCropByPosNegLabeld(
                    keys=["image", "label"],
                    label_key="label",
                    spatial_size=(128, 128, 128),
                    pos=1,
                    neg=1,
                    num_samples=2,
                    image_key="image",
                    image_threshold=0
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[0],
                    prob=0.10
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[1],
                    prob=0.10
                ),
                RandFlipd(
                    keys=["image", "label"],
                    spatial_axis=[2],
                    prob=0.10
                ),
                RandRotate90d(
                    keys=["image", "label"],
                    prob=0.10,
                    max_k=3
                ),
                RandAffined(
                    keys=['image', 'label'],
                    mode=('bilinear', 'nearest'),
                    prob=0.2,
                    spatial_size=(128, 128, 128),
                    rotate_range=(0, 0, np.pi / 15),
                    scale_range=(0.1, 0.1, 0.1)
                ),
                Rand3DElasticd(
                    keys=["image", "label"],
                    prob=0.1,
                    sigma_range=(5, 8),
                    magnitude_range=(100, 200),
                    mode=("bilinear", "nearest")
                ),
                RandBiasFieldd(
                    keys=["image"],
                    prob=0.1,
                    coeff_range=(0.0, 0.1),
                    degree=3
                ),
                RandGaussianNoised(
                    keys=["image"],
                    prob=0.15,
                    mean=0.0,
                    std=0.1
                ),
                RandGaussianSmoothd(
                    keys=["image"],
                    prob=0.15,
                    sigma_x=(0.5, 1.5)
                ),
                RandAdjustContrastd(
                    keys=["image"],
                    prob=0.15,
                    gamma=(0.7, 1.3)
                ),
                RandZoomd(
                    keys=["image", "label"],
                    prob=0.15,
                    min_zoom=0.9,
                    max_zoom=1.1,
                    mode=("trilinear", "nearest")
                ),
                RandShiftIntensityd(
                    keys=["image"],
                    offsets=0.10,
                    prob=0.50
                ),
                ToTensord(keys=["image", "label"])
            ]
            return Compose(train_transforms)

        else:
            # 测试时不使用数据增强
            val_transforms = base_transforms + [
                ToTensord(keys=["image", "label"])
            ]
            return Compose(val_transforms)
    
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
    

def main():
    """
    主函数，用于测试数据加载器和数据处理流程，包括数据加载、数据增强、数据预处理等步骤。
    此部分主要用于后续针对据加载器和数据处理流程，进行调试和优化时使用，平时不会被触发和调用。
    """
    # 数据集路径
    data_dir = "./BraTS2024-BraTS-GLI"  # 替换为实际数据路径

    # 创建数据加载器
    dataset_loader = DatasetLoader(
        data_dir=data_dir,
        cache_rate=0.5,  # 缓存50%的数据
        num_workers=0,   # Windows上设置为0避免多进程问题
        seed=42
    )

    # 获取数据加载器
    train_loader, val_loader = dataset_loader.get_dataloaders(batch_size=1)

    print("\n=== 训练数据信息 ===")
    dataset_loader.print_data_info(train_loader)

    print("\n=== 测试数据信息 ===")
    dataset_loader.print_data_info(val_loader)

    # 演示数据迭代
    print("\n=== 数据迭代信息 ===")
    for i, batch in enumerate(train_loader):

        print(f"\n批次 {i+1}:")
        print(f"图像形状: {batch['image'].shape}")
        print(f"标签形状: {batch['label'].shape}")

        # 检查数据是否包含NaN或无穷值
        if torch.isnan(batch['image']).any():
            print("警告: 图像数据包含NaN值")
        if torch.isinf(batch['image']).any():
            print("警告: 图像数据包含无穷值")

    print("\n数据加载和处理流程完成！")


if __name__ == "__main__":
    main()