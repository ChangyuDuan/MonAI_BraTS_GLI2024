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
    RandBiasFieldd
)
from monai.utils import set_determinism

class MSMultiSpineDatasetLoader:
    """
    MS_MultiSpine数据集加载器
    支持双模态MRI图像和病变分割标签
    
    特点：
    - 自动检测文件命名模式和前缀
    - 支持任意编号范围和文件前缀组合
    - 兼容多种MRI序列类型（T2 + STIR/PSIR/MP2RAGE等）
    - 灵活适应数据集扩展
    """
    
    def __init__(self, 
                 data_dir: str,
                 cache_rate: float = 1.0,
                 num_workers: int = 0,  # Windows上默认设置为0避免多进程问题
                 seed: int = 42):
        """
        初始化MS_MultiSpine数据加载器
        data_dir: 数据集根目录 (preprocessedAndRegistered)
        cache_rate: 缓存比例 (0.0-1.0)
        num_workers: 数据加载工作进程数
        seed: 随机种子
        """
        self.data_dir = data_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        
        # 设置随机种子
        set_determinism(seed=seed)
        
        # 定义图像模态（第二模态根据subject范围动态确定）
        self.modalities = ['T2', 'STIR/PSIR/MP2RAGE']
        
    def _detect_file_pattern(self, case_path: str, case_id: str) -> Optional[Tuple[str, str]]:
        """
        自动检测文件模式，返回(prefix, second_modality)
        通过扫描实际文件来确定前缀和模态类型
        """
        try:
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
    
    def get_transforms(self, mode: str = "train") -> Compose:
        """
        获取数据变换流程
        适配脊柱MRI的特点
        """
        # 基础变换（训练和验证都使用）
        base_transforms = [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            # 注意：MS_MultiSpine不需要标签重映射，直接使用原始标签
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.0, 1.0, 1.0),
                mode=("bilinear", "nearest")
            ),
            # 调整脊柱MRI的强度范围
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-200,  # 适合脊柱MRI的强度范围
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
        
        # 创建数据加载器
        train_loader = MonaiDataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True
        )
        
        val_loader = MonaiDataLoader(
            val_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True
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


def main():
    """
    主函数，用于测试MS_MultiSpine数据加载器
    支持三种不同的模态组合：
    - sub-001到sub-050: T2 + STIR
    - sub-051到sub-075: T2 + PSIR
    - sub-076到sub-100: T2 + MP2RAGE
    """
    # MS_MultiSpine数据集路径
    data_dir = "./MS_MultiSpine_dataset/MS_MultiSpine_dataset/derivatives/preprocessedAndRegistered"

    # 创建数据加载器
    dataset_loader = MSMultiSpineDatasetLoader(
        data_dir=data_dir,
        cache_rate=0.5,  # 缓存50%的数据
        num_workers=0,   # Windows上设置为0避免多进程问题
        seed=42
    )

    # 获取数据加载器
    train_loader, val_loader = dataset_loader.get_dataloaders(batch_size=1)

    print("\n=== MS_MultiSpine训练数据信息 ===")
    dataset_loader.print_data_info(train_loader)

    print("\n=== MS_MultiSpine验证数据信息 ===")
    dataset_loader.print_data_info(val_loader)

    # 演示数据迭代
    print("\n=== 数据迭代信息 ===")
    for i, batch in enumerate(train_loader):
        print(f"\n批次 {i+1}:")
        print(f"图像形状: {batch['image'].shape}")
        print(f"标签形状: {batch['label'].shape}")
        print(f"受试者ID: {batch['subject_id']}")

        # 检查数据是否包含NaN或无穷值
        if torch.isnan(batch['image']).any():
            print("警告: 图像数据包含NaN值")
        if torch.isinf(batch['image']).any():
            print("警告: 图像数据包含无穷值")
        
        # 只显示前3个批次
        if i >= 2:
            break

    print("\nMS_MultiSpine数据加载和处理流程完成！")


if __name__ == "__main__":
    main()