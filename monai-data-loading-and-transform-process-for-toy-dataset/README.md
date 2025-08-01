# BraTS脑肿瘤分割项目

这是一个基于MONAI框架的BraTS脑肿瘤分割项目，支持多种深度学习模型和集成学习方法。项目提供了完整的数据处理、模型训练、评估和可视化功能，内置了丰富的评估指标，包括Dice系数、Hausdorff距离、表面距离、混淆矩阵、平均IoU和广义Dice分数等。

## 🎯 核心特性

### 统一策略架构
项目采用统一的策略架构，确保在训练、评估和部署的所有阶段都使用一致的配置：

- **自适应损失函数策略** - 所有模型默认使用adaptive_combined策略，动态结合多种损失函数
- **完整评估指标** - 统一使用全部6种评估指标进行全面性能评估
- **一致性保证** - 训练-评估-部署全流程配置统一，避免性能差异

### 自适应损失函数
项目使用智能的自适应损失函数策略，根据训练进度动态调整损失权重：

- **DiceCE Loss** - 结合Dice损失和交叉熵损失
- **Focal Loss** - 处理类别不平衡问题
- **Tversky Loss** - 可调节假阳性和假阴性权重
- **动态权重调整** - 根据训练epoch自动调整各损失函数权重

## 项目结构

```
├── main.py                      # 主程序入口
├── model.py                     # 模型定义和创建
├── DatasetLoader_transforms.py  # 数据加载和预处理
├── train.py                     # 训练模块
├── evaluate.py                  # 评估模块
├── inference.py                 # 推理模块
└── utils.py                     # 工具函数
```

## 环境要求

- Python 3.8+
- PyTorch 1.12+
- MONAI 1.0+
- CUDA 11.0+ (可选，用于GPU加速)

### 依赖包安装

```bash
pip install torch torchvision torchaudio
pip install monai[all]
pip install matplotlib pandas tqdm tensorboard
pip install nibabel scikit-image
```

## 快速开始

### 1. 训练模型

项目支持三大类模型训练：**基础模型**、**高级模型**和**集成模型**。每种类型都有详细的训练脚本示例。

#### 🔥 基础模型训练

基础模型是单个深度学习架构的训练，适合快速验证和基准测试。

```bash
# 使用默认配置训练UNet模型
python main.py --mode train --data_dir /path/to/BraTS_data

# 训练特定模型
python main.py --mode train --model_names SegResNet --epochs 100 --batch_size 4 --data_dir /path/to/BraTS_data

# 训练UNETR模型（Transformer架构）
python main.py --mode train --model_names UNETR --epochs 150 --learning_rate 1e-4 --data_dir /path/to/BraTS_data

# 训练SwinUNETR模型（Swin Transformer）
python main.py --mode train --model_names SwinUNETR --epochs 200 --batch_size 2 --data_dir /path/to/BraTS_data

# 训练AttentionUNet模型（注意力机制）
python main.py --mode train --model_names AttentionUNet --epochs 120 --data_dir /path/to/BraTS_data

# 训练VNet模型（3D卷积）
python main.py --mode train --model_names VNet --epochs 100 --batch_size 3 --data_dir /path/to/BraTS_data

# 训练HighResNet模型（高分辨率）
python main.py --mode train --model_names HighResNet --epochs 150 --data_dir /path/to/BraTS_data
```

#### 🚀 高级模型训练

高级模型包括融合网络、知识蒸馏和神经架构搜索，提供更强大的性能。

##### 融合网络训练

融合网络在特征级别结合多个不同架构，通过注意力机制提高性能。

```bash
# 基础融合网络训练（默认使用UNet和SegResNet）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type fusion \
    --epochs 200 --batch_size 2

# 自定义融合模型组合
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type fusion \
    --fusion_models UNet SegResNet UNETR \
    --epochs 250 --learning_rate 5e-5

# 完整融合网络（所有7个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type fusion \
    --fusion_models UNet SegResNet UNETR SwinUNETR AttentionUNet VNet HighResNet \
    --epochs 300 --batch_size 1

# 融合网络高级配置
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type fusion \
    --fusion_models UNet SegResNet AttentionUNet \
    --fusion_type cross_attention \
    --fusion_channels 64 128 256 512 \
    --epochs 200
```

##### 知识蒸馏训练

知识蒸馏使用多个教师模型训练轻量级学生模型，在保持性能的同时减少模型复杂度。

```bash
# 基础知识蒸馏（默认使用所有7个模型作为教师）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type distillation \
    --student_model UNet \
    --epochs 150

# 自定义教师模型的知识蒸馏
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type distillation \
    --teacher_models UNETR SwinUNETR SegResNet \
    --student_model UNet \
    --distillation_temperature 4.0 \
    --distillation_alpha 0.7 \
    --epochs 200

# 多教师蒸馏高级配置
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type distillation \
    --teacher_models UNet SegResNet UNETR SwinUNETR AttentionUNet \
    --student_model UNet \
    --distillation_type multi_teacher \
    --distillation_temperature 5.0 \
    --distillation_alpha 0.8 \
    --epochs 250

# 渐进式知识蒸馏
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type distillation \
    --teacher_models UNETR SwinUNETR \
    --student_model UNet \
    --distillation_type progressive \
    --progressive_stages 3 \
    --epochs 180
```

##### 神经架构搜索训练

神经架构搜索自动发现最优网络架构，减少人工设计需求。

```bash
# 基础NAS训练
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type nas \
    --nas_epochs 50 \
    --epochs 300 --batch_size 1

# DARTS架构搜索
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type nas \
    --nas_type searcher \
    --nas_epochs 80 \
    --arch_lr 3e-4 \
    --model_lr 1e-3 \
    --epochs 400

# 渐进式NAS
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type nas \
    --nas_type progressive \
    --max_layers 6 \
    --start_layers 2 \
    --nas_epochs 60 \
    --epochs 350

# 超网络训练
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_category advanced --model_type nas \
    --nas_type supernet \
    --base_channels 32 \
    --num_layers 4 \
    --epochs 250
```

#### 🎯 集成模型训练

集成模型训练多个独立模型并自动融合，**只要训练的模型大于一个，就会自动启用集成**。

##### 多模型并行集成训练（推荐）

并行训练同时训练多个独立模型，训练速度快，GPU利用率高。

```bash
# 基础并行集成（2个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_names UNet SegResNet \
    --parallel true --epochs 100

# 中等规模并行集成（3个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_names UNet SegResNet UNETR \
    --parallel true --epochs 150

# 大规模并行集成（5个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_names UNet SegResNet UNETR SwinUNETR AttentionUNet \
    --parallel true --epochs 200

# 完整并行集成（所有7个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --enable_ensemble --parallel true \
    --epochs 250 --batch_size 2

# 自定义并行集成配置
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_names UNet SegResNet UNETR AttentionUNet \
    --parallel true \
    --ensemble_weights 0.3 0.3 0.2 0.2 \
    --epochs 180
```

##### 多模型逐个集成训练（节省内存）

逐个训练依次训练每个模型，内存使用较低，适合资源受限环境。

```bash
# 基础逐个集成（2个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_names UNet SegResNet \
    --parallel false --epochs 120

# 中等规模逐个集成（3个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_names UNet SegResNet UNETR \
    --parallel false --epochs 180

# 完整逐个集成（所有7个模型）
python main.py --mode train --data_dir /path/to/BraTS_data \
    --enable_ensemble --parallel false \
    --epochs 300 --batch_size 4

# 逐个集成高级配置
python main.py --mode train --data_dir /path/to/BraTS_data \
    --model_names UNet SegResNet UNETR SwinUNETR \
    --parallel false \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --epochs 200
```

##### 集成训练模式对比

| 特性 | 并行训练 (parallel=true) | 逐个训练 (parallel=false) |
|------|--------------------------|---------------------------|
| **训练速度** | 快 | 慢 |
| **内存使用** | 高 | 低 |
| **GPU利用率** | 高 | 中等 |
| **模型融合** | 自动（>1个模型） | 自动（>1个模型） |
| **适用场景** | 资源充足环境 | 资源受限环境 |
| **推荐程度** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |

#### 📊 训练策略选择指南

| 模型类型 | 适用场景 | 训练时间 | 内存需求 | 性能表现 |
|----------|----------|----------|----------|----------|
| **基础模型** | 快速验证、基准测试 | 短 | 低 | 良好 |
| **融合网络** | 追求最高性能 | 长 | 高 | 优秀 |
| **知识蒸馏** | 模型压缩、部署优化 | 中等 | 中等 | 良好 |
| **神经架构搜索** | 自动化设计 | 很长 | 高 | 优秀 |
| **并行集成** | 资源充足、快速训练 | 中等 | 高 | 优秀 |
| **逐个集成** | 资源受限、稳定训练 | 长 | 低 | 优秀 |

### 2. 评估模型

项目支持对所有类型模型进行全面评估，包括基础模型、高级模型和集成模型。

#### 🔍 基础模型评估

```bash
# 评估单个基础模型
python main.py --mode eval --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/BraTS_data

# 评估特定模型类型
python main.py --mode eval --model_path ./outputs/models/UNet/best_model.pth --data_dir /path/to/BraTS_data

# 评估并保存详细报告
python main.py --mode eval --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/BraTS_data --save_predictions
```

#### 🚀 高级模型评估

```bash
# 评估融合网络模型
python main.py --mode eval --model_path ./outputs/advanced/fusion_model.pth --data_dir /path/to/BraTS_data

# 评估知识蒸馏模型
python main.py --mode eval --model_path ./outputs/advanced/distillation_model.pth --data_dir /path/to/BraTS_data

# 评估NAS搜索模型
python main.py --mode eval --model_path ./outputs/advanced/nas_model.pth --data_dir /path/to/BraTS_data

# 高级模型详细评估
python main.py --mode eval --model_path ./outputs/advanced/fusion_model.pth --data_dir /path/to/BraTS_data \
    --detailed_metrics --save_visualizations
```

#### 🎯 集成模型评估

```bash
# 评估并行集成模型
python main.py --mode eval --model_path ./outputs/ensemble/parallel_ensemble.pth --data_dir /path/to/BraTS_data

# 评估逐个集成模型
python main.py --mode eval --model_path ./outputs/ensemble/sequential_ensemble.pth --data_dir /path/to/BraTS_data

# 评估完整集成模型（7个模型）
python main.py --mode eval --model_path ./outputs/ensemble/full_ensemble.pth --data_dir /path/to/BraTS_data \
    --ensemble_analysis
```

#### 📊 评估参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|----------|
| `--model_path` | str | 必需 | 模型检查点文件路径 |
| `--data_dir` | str | 必需 | 测试数据集路径 |
| `--save_predictions` | flag | False | 保存预测结果 |
| `--detailed_metrics` | flag | False | 计算详细评估指标 |
| `--save_visualizations` | flag | False | 保存可视化结果 |
| `--ensemble_analysis` | flag | False | 集成模型分析 |
| `--batch_size` | int | 1 | 评估批次大小 |

### 3. 模型推理

项目提供强大的推理功能，支持单张图像、批量处理以及各种高级模型的推理。

#### 🖼️ 基础推理

```bash
# 单张图像推理
python main.py --mode inference --model_path ./outputs/checkpoints/best_model.pth --input /path/to/single_image.nii.gz

# 批量推理
python main.py --mode inference --model_path ./outputs/checkpoints/best_model.pth --input /path/to/images_folder --output /path/to/output_folder

# 指定输出格式
python main.py --mode inference --model_path ./outputs/checkpoints/best_model.pth --input /path/to/images \
    --output /path/to/output --save_format nifti
```

#### 🚀 高级模型推理

```bash
# 融合网络推理
python main.py --mode inference --model_path ./outputs/advanced/fusion_model.pth --input /path/to/images \
    --output /path/to/output --model_type fusion

# 知识蒸馏模型推理
python main.py --mode inference --model_path ./outputs/advanced/distillation_model.pth --input /path/to/images \
    --output /path/to/output --model_type distillation

# NAS模型推理
python main.py --mode inference --model_path ./outputs/advanced/nas_model.pth --input /path/to/images \
    --output /path/to/output --model_type nas

# 高级推理配置
python main.py --mode inference --model_path ./outputs/advanced/fusion_model.pth --input /path/to/images \
    --output /path/to/output --batch_size 4 --overlap 0.5 --blend_mode gaussian --tta
```

#### 🎯 集成模型推理

```bash
# 并行集成推理
python main.py --mode inference --model_path ./outputs/ensemble/parallel_ensemble.pth --input /path/to/images \
    --output /path/to/output --ensemble_mode parallel

# 逐个集成推理
python main.py --mode inference --model_path ./outputs/ensemble/sequential_ensemble.pth --input /path/to/images \
    --output /path/to/output --ensemble_mode sequential

# 完整集成推理（7个模型）
python main.py --mode inference --model_path ./outputs/ensemble/full_ensemble.pth --input /path/to/images \
    --output /path/to/output --ensemble_mode full --voting_strategy soft
```

#### ⚡ 高性能推理

```bash
# GPU加速推理
python main.py --mode inference --model_path ./outputs/checkpoints/best_model.pth --input /path/to/images \
    --output /path/to/output --device cuda --batch_size 8

# 混合精度推理
python main.py --mode inference --model_path ./outputs/checkpoints/best_model.pth --input /path/to/images \
    --output /path/to/output --amp --batch_size 16

# 多GPU推理
python main.py --mode inference --model_path ./outputs/checkpoints/best_model.pth --input /path/to/images \
    --output /path/to/output --multi_gpu --batch_size 32
```

#### 📊 推理参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|----------|
| `--model_path` | str | 必需 | 模型检查点文件路径 |
| `--input` | str | 必需 | 输入图像或文件夹路径 |
| `--output` | str | 必需 | 输出结果保存路径 |
| `--model_type` | str | auto | 模型类型（basic/fusion/distillation/nas） |
| `--batch_size` | int | 1 | 推理批次大小 |
| `--overlap` | float | 0.25 | 滑动窗口重叠率 |
| `--blend_mode` | str | constant | 融合模式（constant/gaussian） |
| `--tta` | flag | False | 测试时增强 |
| `--ensemble_mode` | str | None | 集成模式（parallel/sequential/full） |
| `--voting_strategy` | str | soft | 投票策略（soft/hard/weighted） |
| `--save_format` | str | nifti | 输出格式（nifti/png/jpg） |
| `--device` | str | auto | 计算设备（cpu/cuda/auto） |
| `--amp` | flag | False | 混合精度推理 |
| `--multi_gpu` | flag | False | 多GPU推理 |

#### 🎯 推理输出结果

1. **预测结果文件** (`*.nii.gz`)
   - 分割掩码，包含不同的标签值
   - 标签含义：0=背景，1=坏死，2=水肿，3=增强肿瘤
   - 支持多种输出格式（NIfTI、PNG、JPG）

2. **可视化文件** (`*_visualization.png`)
   - 原始图像和预测结果的叠加显示
   - 中间层切片的可视化
   - 高级模型特有的注意力图可视化

3. **推理报告** (`inference_report.json`)
   - 包含所有文件的推理结果统计
   - 模型类型和配置信息
   - 推理时间和性能指标
   - 成功/失败状态和错误信息

4. **集成模型输出** (仅集成模型)
   - 各个子模型的预测结果
   - 投票权重和置信度分数
   - 集成策略分析报告

#### 💻 编程接口使用

```python
from inference import InferenceEngine

# 基础模型推理
engine = InferenceEngine(
    model_path='outputs/checkpoints/best_model.pth',
    device='cuda'
)

# 高级模型推理
advanced_engine = InferenceEngine(
    model_path='outputs/advanced/fusion_model.pth',
    model_type='fusion',
    device='cuda'
)

# 集成模型推理
ensemble_engine = InferenceEngine(
    model_path='outputs/ensemble/full_ensemble.pth',
    ensemble_mode='full',
    voting_strategy='soft'
)

# 单文件推理
result = engine.predict_single_case(
    image_path='data/test.nii.gz',
    output_path='results/prediction.nii.gz',
    tta=True  # 测试时增强
)

# 批量推理
results = engine.predict_batch(
    input_dir='data/test_cases/',
    output_dir='results/',
    batch_size=4
)
```

#### ✨ 推理模块特点

- **全模型支持**: 支持基础模型、高级模型和集成模型
- **智能识别**: 自动识别模型类型和配置
- **高效推理**: 优化的滑动窗口和批处理策略
- **测试时增强**: 支持TTA提升预测精度
- **多GPU加速**: 支持多GPU并行推理
- **混合精度**: 支持AMP加速推理过程
- **灵活输出**: 多种输出格式和可视化选项
- **详细报告**: 完整的推理过程和结果分析

## 支持的模型

项目支持以下7种深度学习模型：

1. **UNet** - 经典的U型网络架构
2. **SegResNet** - 基于ResNet的分割网络
3. **UNETR** - 基于Transformer的U型网络
4. **SwinUNETR** - 基于Swin Transformer的分割网络
5. **AttentionUNet** - 带注意力机制的U型网络
6. **VNet** - 3D卷积分割网络
7. **HighResNet** - 高分辨率网络

## 高级模型功能

本项目支持三种高级模型设计方法，提供更强大的模型性能和灵活性：

- **知识蒸馏 (Knowledge Distillation)**: 使用多个教师模型训练轻量级学生模型
- **融合网络 (Fusion Networks)**: 在特征级别融合多个不同架构的模型
- **神经架构搜索 (Neural Architecture Search)**: 自动搜索最优网络架构

**默认配置**: 所有高级模型训练默认使用全部7个网络架构（UNet、SegResNet、UNETR、SwinUNETR、AttentionUNet、VNet、HighResNet），以获得最佳性能。

### 知识蒸馏

#### 基本概念

知识蒸馏通过让学生模型学习教师模型的"软标签"来提高性能，同时保持模型的轻量化。

#### 使用方法

```bash
# 基本知识蒸馏（默认使用所有7个网络架构作为教师模型）
python main.py train \
    --data_dir ./data/training_data \
    --model_category advanced \
    --model_type distillation \
    --student_model UNet \
    --distillation_temperature 4.0 \
    --distillation_alpha 0.7 \
    --epochs 100

# 自定义教师模型
python main.py train \
    --data_dir ./data/training_data \
    --model_category advanced \
    --model_type distillation \
    --teacher_models UNet SegResNet UNETR \
    --student_model UNet \
    --distillation_temperature 4.0 \
    --distillation_alpha 0.7 \
    --epochs 100
```

#### 参数说明

- `teacher_models`: 教师模型列表，支持多个模型
- `student_model`: 学生模型，通常选择轻量级模型
- `distillation_temperature`: 蒸馏温度，控制软标签的平滑程度
- `distillation_alpha`: 软标签权重，平衡软标签和硬标签的重要性

### 融合网络

#### 基本概念

融合网络在特征级别结合多个不同架构的模型，通过注意力机制和自适应融合提高性能。

#### 使用方法

```bash
# 融合网络训练（默认使用所有7个网络架构）
python main.py train \
    --data_dir ./data/training_data \
    --model_category advanced \
    --model_type fusion \
    --epochs 200

# 自定义融合模型
python main.py train \
    --data_dir ./data/training_data \
    --model_category advanced \
    --model_type fusion \
    --fusion_models UNet SegResNet AttentionUNet \
    --epochs 200
```

#### 特性

- **跨模型注意力**: 不同模型特征之间的交互
- **通道注意力**: 重要特征通道的自适应权重
- **空间注意力**: 重要空间位置的自适应权重
- **自适应融合门**: 动态调整不同模型的贡献

### 神经架构搜索

#### 基本概念

NAS通过自动搜索最优的网络架构，减少人工设计的需要。本实现基于DARTS算法。

#### 使用方法

```bash
# NAS搜索
python main.py train \
    --data_dir ./data/training_data \
    --model_category advanced \
    --model_type nas \
    --nas_epochs 50 \
    --batch_size 1 \
    --epochs 300
```

#### 搜索过程

1. **架构搜索阶段**: 优化架构参数
2. **模型训练阶段**: 使用找到的最优架构训练模型
3. **渐进式搜索**: 逐步增加搜索复杂度

### 高级模型命令行参数

```bash
# 模型类别和类型
--model_category {basic,advanced}     # 模型类别
--model_type {single,fusion,distillation,nas}  # 高级模型类型

# 知识蒸馏参数
--teacher_models MODEL1 MODEL2 ...    # 教师模型列表
--student_model MODEL                  # 学生模型
--distillation_temperature FLOAT      # 蒸馏温度
--distillation_alpha FLOAT            # 软标签权重

# 融合网络参数
--fusion_models MODEL1 MODEL2 ...     # 融合模型列表

# NAS参数
--nas_epochs INT                       # NAS搜索轮数
```

### 高级模型配置示例

#### 知识蒸馏配置

```python
distillation_config = {
    # 默认使用所有7个网络架构作为教师模型
    'teacher_models': ['UNet', 'SegResNet', 'UNETR', 'SwinUNETR', 'AttentionUNet', 'VNet', 'HighResNet'],
    'student_model': 'UNet',                           # 学生模型
    'distillation_temperature': 4.0,                   # 蒸馏温度
    'distillation_alpha': 0.7,                         # 软标签权重
    'progressive_stages': 3,                            # 渐进式阶段数
    'stage_epochs': [30, 30, 40]                       # 每阶段训练轮数
}
```

#### 融合网络配置

```python
fusion_config = {
    # 默认使用所有7个网络架构进行融合
    'fusion_models': ['UNet', 'SegResNet', 'UNETR', 'SwinUNETR', 'AttentionUNet', 'VNet', 'HighResNet'],
    'fusion_dim': 256,                                         # 融合特征维度
    'attention_heads': 8,                                      # 注意力头数
    'dropout_rate': 0.1,                                       # Dropout率
    'use_cross_attention': True,                               # 使用跨模型注意力
    'use_channel_attention': True,                             # 使用通道注意力
    'use_spatial_attention': True                              # 使用空间注意力
}
```

#### NAS配置

```python
nas_config = {
    'nas_epochs': 50,                    # 架构搜索轮数
    'search_space': 'darts',             # 搜索空间类型
    'arch_lr': 3e-4,                     # 架构学习率
    'model_lr': 2e-4,                    # 模型学习率
    'progressive_stages': 3,             # 渐进式搜索阶段
    'operations': [                      # 候选操作
        'conv_3x3', 'conv_5x5', 'dilated_conv',
        'attention', 'skip_connect', 'pool'
    ]
}
```

### 高级模型注意事项

#### 硬件要求

- **GPU内存**: 高级模型需要更多GPU内存，建议至少8GB
- **训练时间**: 比基础模型需要更长的训练时间
- **CPU模式**: 可以运行但速度较慢

#### 性能优化

1. **批次大小**: 根据GPU内存调整，高级模型建议使用较小批次
2. **混合精度**: 启用AMP可以减少内存使用
3. **梯度累积**: 在小批次时使用梯度累积

#### 最佳实践

1. **知识蒸馏**: 
   - 默认使用所有7个网络架构作为教师模型，获得最佳知识转移效果
   - 先训练好教师模型，再进行蒸馏
   - 可根据资源限制自定义教师模型数量

2. **融合网络**: 
   - 默认融合所有7个网络架构，充分利用不同模型的优势
   - 选择互补性强的基础模型可进一步提升性能
   - 注意GPU内存使用，必要时减少融合模型数量

3. **NAS**: 
   - 从小规模搜索空间开始，逐步扩大
   - 利用渐进式搜索策略提高效率

## 评估指标

项目内置了全面的评估指标体系，**所有模型在训练、评估和部署阶段都统一使用完整的6种评估指标**：

### 核心分割指标
- **Dice系数** - 衡量分割重叠度，范围[0,1]，越高越好
- **平均IoU (Intersection over Union)** - 交并比指标，衡量分割准确性
- **广义Dice分数** - 加权Dice指标，处理类别不平衡问题

### 边界和形状指标
- **Hausdorff距离** - 衡量边界准确性，距离越小越好
- **表面距离** - 衡量表面重建质量和边界平滑度

### 分类性能指标
- **混淆矩阵指标** - 详细的分类性能分析，包括精确率、召回率、F1分数等

### 统一评估策略
- **自动计算** - 训练过程中实时计算所有指标
- **多类别支持** - 支持背景、坏死核心、水肿区域、增强肿瘤等多类别评估
- **统计分析** - 提供均值、标准差、最值等统计信息
- **可视化展示** - 生成指标趋势图和分布图

## 命令行参数详解

### 基本参数

- `--mode` - 运行模式，可选值：`train`（训练）、`eval`（评估）
- `--data_dir` - BraTS数据集路径
- `--device` - 计算设备，可选值：`auto`（自动检测）、`cpu`、`cuda`

### 训练参数

#### 基础训练参数

- `--model_name` - 单个模型名称（UNet、SegResNet、UNETR等）
- `--model_names` - 多个模型名称列表
- `--enable_ensemble` - 启用集成学习
- `--epochs` - 训练轮数（默认：50）
- `--batch_size` - 批次大小（默认：自动调整）
- `--learning_rate` - 学习率（默认：1e-4）
- `--output_dir` - 输出目录（默认：./outputs）

#### 高级模型参数

- `--model_category` - 模型类别，可选值：`basic`（基础模型）、`advanced`（高级模型）
- `--model_type` - 高级模型类型，可选值：`single`、`fusion`、`distillation`、`nas`

##### 知识蒸馏参数

- `--teacher_models` - 教师模型列表（默认：所有7个网络架构）
- `--student_model` - 学生模型名称（默认：UNet）
- `--distillation_temperature` - 蒸馏温度（默认：4.0）
- `--distillation_alpha` - 软标签权重（默认：0.7）

##### 融合网络参数

- `--fusion_models` - 融合模型列表（默认：所有7个网络架构）

##### NAS参数

- `--nas_epochs` - NAS搜索轮数（默认：50）

### 评估参数

- `--model_path` - 模型文件路径
- `--output_dir` - 评估结果输出目录（默认：./evaluation_results）

## 使用示例

### 基础训练

```bash
# 训练单个UNet模型
python main.py --mode train --model_name UNet --data_dir /path/to/BraTS --epochs 100

# 训练SegResNet模型，使用GPU
python main.py --mode train --model_name SegResNet --device cuda --epochs 150 --batch_size 2

# 训练UNETR模型，自定义学习率
python main.py --mode train --model_name UNETR --learning_rate 5e-5 --epochs 200
```

### 多模型训练

```bash
# 顺序训练多个模型
python main.py --mode train --model_names UNet SegResNet AttentionUNet --epochs 50

# 训练所有支持的模型
python main.py --mode train --model_names UNet SegResNet UNETR SwinUNETR AttentionUNet VNet HighResNet --epochs 30
```

### 集成学习

```bash
# 启用集成学习（自动选择所有支持的模型）
python main.py --mode train --enable_ensemble --epochs 100

# 集成学习结合多模型训练
python main.py --mode train --model_names UNet SegResNet UNETR --enable_ensemble --epochs 80
```

### 高级模型训练

#### 知识蒸馏训练

```bash
# 基本知识蒸馏（默认使用所有7个网络架构作为教师模型）
python main.py --mode train \
    --data_dir /path/to/BraTS \
    --model_category advanced \
    --model_type distillation \
    --student_model UNet \
    --epochs 100

# 自定义教师模型的知识蒸馏
python main.py --mode train \
    --data_dir /path/to/BraTS \
    --model_category advanced \
    --model_type distillation \
    --teacher_models UNet SegResNet UNETR \
    --student_model UNet \
    --distillation_temperature 4.0 \
    --distillation_alpha 0.7 \
    --epochs 100
```

#### 融合网络训练

```bash
# 融合网络训练（默认使用所有7个网络架构）
python main.py --mode train \
    --data_dir /path/to/BraTS \
    --model_category advanced \
    --model_type fusion \
    --epochs 200

# 自定义融合模型
python main.py --mode train \
    --data_dir /path/to/BraTS \
    --model_category advanced \
    --model_type fusion \
    --fusion_models UNet SegResNet AttentionUNet \
    --epochs 200
```

#### 神经架构搜索训练

```bash
# NAS搜索训练
python main.py --mode train \
    --data_dir /path/to/BraTS \
    --model_category advanced \
    --model_type nas \
    --nas_epochs 50 \
    --batch_size 1 \
    --epochs 300

# 自定义NAS参数
python main.py --mode train \
    --data_dir /path/to/BraTS \
    --model_category advanced \
    --model_type nas \
    --nas_epochs 100 \
    --batch_size 1 \
    --epochs 500 \
    --learning_rate 2e-4
```

### 模型评估

```bash
# 基础评估
python main.py --mode eval --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/BraTS

# 评估集成模型
python main.py --mode eval --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/BraTS

# 指定输出目录评估
python main.py --mode eval --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/BraTS --output_dir ./my_evaluation
```

### 模型推理

```bash
# 单个文件推理
python main.py --mode inference \
    --model_path outputs/checkpoints/best_model.pth \
    --input data/test_case.nii.gz \
    --output results/prediction.nii.gz

# 批量推理
python main.py --mode inference \
    --model_path outputs/checkpoints/best_model.pth \
    --input data/test_cases/ \
    --output results/ \
    --batch_inference

# 高级推理配置
python main.py --mode inference \
    --model_path outputs/checkpoints/best_model.pth \
    --input data/test.nii.gz \
    --output results/pred.nii.gz \
    --device cuda \
    --roi_size 128 128 128 \
    --sw_batch_size 4 \
    --overlap 0.5

# GPU加速推理
python main.py --mode inference \
    --device cuda \
    --sw_batch_size 8 \
    --model_path outputs/checkpoints/best_model.pth \
    --input data/test.nii.gz \
    --output results/prediction.nii.gz

# 批量处理优化（禁用可视化）
python main.py --mode inference \
    --batch_inference \
    --no_visualization \
    --model_path outputs/checkpoints/best_model.pth \
    --input data/test_cases/ \
    --output results/
```

## 项目特性

### 🎯 统一策略架构

- **全流程一致性** - 训练、评估、部署使用统一的损失函数和评估指标
- **自适应损失函数** - 智能组合多种损失函数，动态调整权重
- **完整评估体系** - 统一使用6种评估指标，全面评估模型性能
- **配置统一管理** - 所有模型实例自动应用统一策略配置

### 🚀 高性能优化

- **自动设备检测** - 智能选择CPU或GPU
- **内存优化** - 根据设备自动调整批次大小和缓存策略
- **多进程数据加载** - 加速数据预处理
- **滑动窗口推理** - 支持大尺寸图像分割

### 🎯 智能训练

- **自适应学习率** - 使用余弦退火调度器
- **动态损失调整** - 根据训练进度自动调整损失函数权重
- **早停机制** - 防止过拟合
- **模型检查点** - 自动保存最佳模型
- **训练监控** - 实时显示训练进度和所有评估指标

### 📊 全面评估

- **统一评估标准** - 所有模型使用相同的6种评估指标
- **多维度分析** - 从分割精度、边界质量、分类性能等多角度评估
- **可视化结果** - 生成分割结果对比图和指标趋势图
- **详细报告** - 输出完整的评估报告和统计分析
- **性能对比** - 支持多模型性能对比分析

### 🔧 易用性

- **一键训练** - 简单的命令行接口，自动应用最佳配置
- **智能配置** - 自动应用统一策略，无需手动设置
- **错误处理** - 友好的错误提示和配置验证
- **进度显示** - 清晰的训练和评估进度，实时指标监控

## 输出结果

### 训练输出

训练完成后，会在输出目录生成以下文件：

```
outputs/
├── checkpoints/
│   ├── best_model.pth              # 最佳模型权重（单模型或集成模型）
│   └── latest_model.pth            # 最新模型权重
├── logs/
│   └── tensorboard/                # TensorBoard日志
├── metrics/
│   ├── training_history.json       # 训练历史记录
│   └── training_curves.png         # 训练曲线图
└── visualizations/
    └── sample_predictions.png      # 样本预测可视化
```

### 评估输出

评估完成后，会生成详细的评估报告：

```
evaluation_results/
├── case_results.csv           # 每个案例的详细结果
├── summary_results.txt        # 总体统计结果
├── results_distribution.png   # 结果分布图
└── visualizations/
    ├── case_001_prediction.png
    ├── case_002_prediction.png
    └── ...
```

## 数据格式要求

BraTS数据集应按以下结构组织：

```
BraTS_data/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t1n.nii.gz    # T1 native
│   ├── BraTS-GLI-00000-000-t1c.nii.gz    # T1 contrast-enhanced
│   ├── BraTS-GLI-00000-000-t2w.nii.gz    # T2 weighted
│   ├── BraTS-GLI-00000-000-t2f.nii.gz    # T2 FLAIR
│   └── BraTS-GLI-00000-000-seg.nii.gz    # 分割标注（训练时需要）
├── BraTS-GLI-00001-000/
│   ├── BraTS-GLI-00001-000-t1n.nii.gz
│   ├── BraTS-GLI-00001-000-t1c.nii.gz
│   ├── BraTS-GLI-00001-000-t2w.nii.gz
│   ├── BraTS-GLI-00001-000-t2f.nii.gz
│   └── BraTS-GLI-00001-000-seg.nii.gz
└── ...
```

**注意**: 
- 每个案例目录名称应与文件前缀保持一致
- 支持的模态：t1n（T1 native）、t1c（T1 contrast-enhanced）、t2w（T2 weighted）、t2f（T2 FLAIR）
- 分割文件（seg.nii.gz）在训练时必需，评估时可选

## 🔧 统一策略技术实现

### 自适应损失函数策略

项目实现了智能的自适应损失函数组合策略，确保所有模型都使用最优的损失函数配置：

#### 损失函数组合
```python
# 自适应损失函数策略 (adaptive_combined)
loss_functions = {
    'dice_ce': DiceCELoss(to_onehot_y=True, softmax=True),
    'focal': FocalLoss(to_onehot_y=True, use_softmax=True),
    'tversky': TverskyLoss(to_onehot_y=True, softmax=True)
}
```

#### 动态权重调整
- **早期训练** (epoch < 30%): 重点使用Dice+CE损失，权重[0.6, 0.2, 0.2]
- **中期训练** (30% ≤ epoch < 70%): 平衡所有损失函数，权重[0.4, 0.3, 0.3]
- **后期训练** (epoch ≥ 70%): 强化Tversky损失，权重[0.3, 0.2, 0.5]

### 完整评估指标体系

所有模型统一使用6种评估指标，确保评估的全面性和一致性：

#### 指标配置
```python
metrics = {
    'dice': DiceMetric(include_background=False, reduction="mean"),
    'hausdorff': HausdorffDistanceMetric(include_background=False, reduction="mean"),
    'surface_distance': SurfaceDistanceMetric(include_background=False, reduction="mean"),
    'confusion_matrix': ConfusionMatrixMetric(metric_name="accuracy", reduction="mean"),
    'mean_iou': MeanIoU(include_background=False, reduction="mean"),
    'generalized_dice': GeneralizedDiceScore(include_background=False, reduction="mean")
}
```

### 统一配置管理

#### 全局配置函数
- `get_high_performance_config()`: 返回包含统一策略的配置
- 所有模型实例自动应用 `use_adaptive_loss=True` 和 `use_full_metrics=True`
- 训练、评估、部署阶段配置完全一致

#### 配置验证
- 自动验证损失函数和评估指标配置
- 确保所有模型实例使用统一策略
- 提供详细的配置日志输出

### 技术优势

1. **性能一致性**: 消除训练-评估-部署阶段的性能差异
2. **智能优化**: 自适应损失函数根据训练进度动态优化
3. **全面评估**: 6种指标从多个维度全面评估模型性能
4. **易于维护**: 统一配置管理，减少配置错误和不一致
5. **可扩展性**: 新增模型自动继承统一策略配置

## 详细文件说明

### main.py - 主程序入口

这是项目的主要入口文件，负责解析命令行参数、配置管理和调用相应的训练或评估功能。

#### 主要函数详解：

**第1-20行：导入和文档说明**
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BraTS脑肿瘤分割项目主程序

这是一个基于MONAI框架的BraTS脑肿瘤分割项目，支持多种深度学习模型。
项目提供了完整的数据处理、模型训练、评估和可视化功能。

使用示例:
    python main.py --mode train --data_dir /path/to/dataset
    python main.py --mode eval --model_path /path/to/model.pth --data_dir /path/to/dataset
    python main.py --mode train --model_name UNet --epochs 100 --batch_size 4
    python main.py --mode train --enable_ensemble --epochs 200

作者: 个人使用版本
版本: 3.1.0
"""
```

**第21-35行：模块导入**
```python
import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Dict, Any, List

# 导入项目模块
from train import ModelTrainer
from evaluate import BraTSEvaluator
from model import get_all_supported_models
from utils import format_time
```
- 导入标准库和第三方库
- 导入项目自定义模块：训练器、评估器、模型工具和实用函数

**第36-85行：get_high_performance_config函数**
```python
def get_high_performance_config(device_type: str = "auto") -> Dict[str, Any]:
    """
    获取高性能配置
    
    Args:
        device_type: 设备类型 ('cpu', 'cuda', 'auto')
        
    Returns:
        配置字典
    """
```
- 根据设备类型（CPU/GPU）返回优化的配置参数
- CPU配置：较小的批次大小(1)、缓存率(0.1)、工作进程数(2)
- GPU配置：较大的批次大小(4)、缓存率(0.5)、工作进程数(4)
- 包含数据、模型、训练等各方面的配置参数

**第86-108行：merge_args_with_config函数**
```python
def merge_args_with_config(args: argparse.Namespace, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    合并命令行参数和配置字典
    
    Args:
        args: 命令行参数
        config: 基础配置字典
        
    Returns:
        合并后的配置字典
    """
```
- 将命令行参数覆盖到基础配置中
- 处理特殊参数如模型列表、集成模式等
- 确保参数的正确性和一致性

**第109-200行：run_simplified_training函数**
```python
def run_simplified_training(config: Dict[str, Any]) -> None:
    """
    运行简化的训练流程，支持多模型训练和集成
    
    Args:
        config: 训练配置字典
    """
```
- 支持单模型和集成模型训练
- 处理多模型顺序训练
- 集成模型的创建和训练
- 输出训练结果和模型信息

**第201-250行：auto_adjust_parameters函数**
```python
def auto_adjust_parameters(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据设备和内存自动调节参数
    
    Args:
        config: 原始配置
        
    Returns:
        调节后的配置
    """
```
- 根据GPU内存自动调整批次大小
- 根据CPU核心数调整工作进程数
- 优化缓存率和其他性能参数

**第251-350行：run_evaluation函数**
```python
def run_evaluation(config: Dict[str, Any]) -> None:
    """
    运行模型评估
    
    Args:
        config: 评估配置字典
    """
```
- 创建评估器实例
- 执行模型评估
- 生成评估报告和可视化结果
- 输出详细的性能指标

**第351-550行：main函数**
```python
def main():
    """
    主函数：解析命令行参数并执行相应操作
    """
    parser = argparse.ArgumentParser(
        description="BraTS脑肿瘤分割项目",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  训练模型:
    python main.py --mode train --data_dir /path/to/dataset
    python main.py --mode train --model_name UNet --epochs 100
    
  评估模型:
    python main.py --mode eval --model_path /path/to/model.pth --data_dir /path/to/dataset
        """
    )
```
- 定义所有命令行参数
- 包括模式选择、数据路径、模型配置、训练参数等
- 参数验证和设备配置
- 根据模式调用相应的训练或评估函数

**第551-589行：程序入口和设备配置**
```python
    # 设备配置
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    config['device'] = device
    
    # 打印欢迎信息
    print("=" * 60)
    print("🧠 BraTS脑肿瘤分割项目 v3.1.0")
    print("=" * 60)
    print(f"运行模式: {args.mode}")
    print(f"设备: {device}")
    print(f"数据目录: {args.data_dir}")
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"⚠️  警告: 数据目录不存在: {args.data_dir}")
        print("请确保数据路径正确")
    
    # 根据模式执行相应操作
    if args.mode == 'train':
        run_simplified_training(config)
    elif args.mode == 'eval':
        run_evaluation(config)
    else:
        print(f"❌ 不支持的模式: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()
```
- 自动检测和配置计算设备
- 打印项目信息和运行参数
- 验证数据目录存在性
- 根据模式调用训练或评估功能

### model.py - 模型定义和创建

这个文件定义了所有支持的深度学习模型和相关的创建函数。

#### 主要类和函数详解：

**第1-15行：导入声明**
```python
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple, Any, Optional
from monai.networks.nets import UNet, SegResNet, UNETR, SwinUNETR, AttentionUnet, VNet, HighResNet
from monai.networks.layers import Norm
from monai.losses import DiceCELoss, FocalLoss, TverskyLoss, GeneralizedDiceLoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric, SurfaceDistanceMetric, ConfusionMatrixMetric, MeanIoU, GeneralizedDiceScore
from monai.transforms import Compose, Activations, AsDiscrete
from monai.inferers import sliding_window_inference

import torch.nn.functional as F
```
- 导入PyTorch核心模块
- 导入MONAI的网络架构、损失函数、评估指标
- 导入推理和变换工具

**第18-40行：BasicModelBank类初始化**
```python
class BasicModelBank:
    """
    简化的BraTS分割模型，支持基础模型架构
    """
    def __init__(self, model_name: str = 'UNet', device: str = 'auto'):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = self._create_model()
        self.loss_function = self._create_loss()
        self.metrics = self._create_metrics()
```
- 初始化模型名称和设备
- 创建模型、损失函数和评估指标
- 支持自动设备检测

**第41-55行：设备设置方法**
```python
def _setup_device(self, device: str) -> torch.device:
    """设置设备"""
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
```
- 智能设备选择逻辑
- 自动检测CUDA可用性
- 提供设备切换的安全机制

**第56-150行：模型创建方法**
```python
def _create_model(self):
    """创建模型"""
    if self.model_name == 'UNet':
        model = UNet(
            spatial_dims=3,
            in_channels=4,
            out_channels=4,
            channels=(32, 64, 128, 256, 512),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="instance",
            dropout=0.1
        )
```
- 支持7种不同的网络架构：UNet、SegResNet、UNETR、SwinUNETR、AttentionUNet、VNet、HighResNet
- 每个模型都针对BraTS数据集进行了优化配置
- 统一的输入输出接口：4个输入通道（T1、T1ce、T2、FLAIR），4个输出类别

**第151-170行：损失函数创建**
```python
def _create_loss(self):
    """创建损失函数"""
    return DiceCELoss(
        to_onehot_y=True,
        softmax=True,
        squared_pred=True,
        jaccard=False,
        reduction="mean"
    )
```
- 使用Dice和交叉熵的组合损失
- 适合医学图像分割任务
- 自动处理one-hot编码和softmax激活

**第171-190行：评估指标创建**
```python
def _create_metrics(self):
    """创建评估指标"""
    return {
        'dice': DiceMetric(
            include_background=False,
            reduction="mean_batch",
            get_not_nans=False
        ),
        'hausdorff': HausdorffDistanceMetric(
            include_background=False,
            distance_metric='euclidean',
            percentile=95,
            directed=False,
            reduction="mean_batch"
        )
    }
```
- 创建Dice系数和Hausdorff距离指标
- 排除背景类别的计算
- 使用批次平均的约简方式

**第191-220行：滑动窗口推理**
```python
def sliding_window_inference(self, inputs: torch.Tensor, 
                            roi_size: Tuple = (128, 128, 128),
                            sw_batch_size: int = 4,
                            overlap: float = 0.6) -> torch.Tensor:
    """滑动窗口推理"""
    return sliding_window_inference(
        inputs=inputs,
        roi_size=roi_size,
        sw_batch_size=sw_batch_size,
        predictor=self.model,
        overlap=overlap,
        mode="gaussian",
        sw_device=self.device,
        device=self.device
    )
```
- 实现滑动窗口推理策略
- 支持大尺寸图像的分块处理
- 使用高斯权重融合重叠区域

**第221-280行：BankModelIntegration类**
```python
class BankModelIntegration:
    """集成模型，支持多模型投票"""
    def __init__(self, models: List[BasicModelBank], device: str = 'auto'):
        self.models = models
        self.device = self._setup_device(device)
        
    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        """集成预测（平均）"""
        predictions = []
        for model in self.models:
            with torch.no_grad():
                pred = model.sliding_window_inference(inputs)
                predictions.append(pred)
        
        # 平均融合
        ensemble_pred = torch.mean(torch.stack(predictions), dim=0)
        return ensemble_pred
```
- 实现多模型集成预测
- 支持平均融合和投票融合两种策略
- 提高模型的鲁棒性和准确性

**第281-320行：优化器创建函数**
```python
def create_optimizer(model: nn.Module, 
                    optimizer_name: str = "AdamW",
                    learning_rate: float = 1e-4,
                    weight_decay: float = 1e-5) -> optim.Optimizer:
    """创建优化器"""
    if optimizer_name.lower() == "adam":
        return optim.Adam(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
```
- 支持Adam、AdamW、SGD三种优化器
- 提供统一的参数接口
- 默认使用AdamW优化器

**第321-360行：学习率调度器创建函数**
```python
def create_scheduler(optimizer: optim.Optimizer,
                    scheduler_name: str = "CosineAnnealingLR",
                    **kwargs):
    """创建学习率调度器"""
    if scheduler_name.lower() == "steplr":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1)
        )
```
- 支持StepLR、CosineAnnealingLR、ReduceLROnPlateau调度器
- 灵活的参数配置
- 默认使用余弦退火调度

### DatasetLoader_transforms.py - 数据加载和预处理

这个文件负责BraTS数据集的加载、预处理和数据增强。

#### 主要类和函数详解：

**第1-30行：导入声明**
```python
import os
import glob
import numpy as np
import nibabel as nib
from typing import Dict, List, Tuple, Optional

import torch
from torch.utils.data import DataLoader

from monai.data import Dataset, CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd,
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld,
    RandFlipd, RandRotate90d, RandShiftIntensityd, RandAffined,
    ToTensord, EnsureTyped, Resized, NormalizeIntensityd,
    RandGaussianNoised, RandGaussianSmoothd, RandAdjustContrastd,
    RandZoomd, Rand3DElasticd, RandBiasFieldd
)
from monai.utils import set_determinism
```
- 导入数据处理相关的库
- 导入MONAI的数据加载和变换工具
- 导入各种数据增强变换

**第31-55行：DatasetLoader类初始化**
```python
class DatasetLoader:
    """
    BraTS2024-BraTS-GLI数据集加载器
    支持多模态MRI图像（T1, T1ce, T2, FLAIR）和分割标签
    """
    
    def __init__(self, 
                 data_dir: str,
                 cache_rate: float = 1.0,
                 num_workers: int = 4,
                 seed: int = 42):
        self.data_dir = data_dir
        self.cache_rate = cache_rate
        self.num_workers = num_workers
        
        # 设置随机种子
        set_determinism(seed=seed)
        
        # 定义图像模态
        self.modalities = ['t1n', 't1c', 't2f', 't2w']
```
- 初始化数据目录和加载参数
- 设置随机种子确保可重复性
- 定义BraTS数据集的四种MRI模态

**第56-120行：数据字典获取方法**
```python
def get_data_dicts(self) -> Tuple[List[Dict], List[Dict]]:
    """获取训练和验证数据字典"""
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
```
- 自动扫描数据目录
- 构建标准的BraTS文件路径
- 检查所有必需文件的存在性
- 按8:2比例划分训练和验证集

**第121-200行：数据变换获取方法**
```python
def get_transforms(self, mode: str = "train") -> Compose:
    """获取数据变换流程"""
    # 基础变换（训练和验证都使用）
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
```
- 定义基础的数据预处理流程
- 包括加载、方向统一、重采样、强度归一化等
- 裁剪前景区域并调整到统一尺寸

**第201-280行：训练时数据增强**
```python
if mode == "train":
    # 训练时添加数据增强
    train_transforms = base_transforms + [
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(96, 96, 96),
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
        # ... 更多数据增强变换
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
        )
    ]
```
- 丰富的数据增强策略
- 包括几何变换、强度变换、噪声添加等
- 专门针对医学图像的增强方法

**第281-350行：数据加载器创建**
```python
def get_dataloaders(self, batch_size: int = 2) -> Tuple[DataLoader, DataLoader]:
    """获取训练和验证数据加载器"""
    # 获取数据文件列表
    train_files, val_files = self.get_data_dicts()
    
    return self.create_dataloaders_from_dicts(train_files, val_files, batch_size)

def create_dataloaders_from_dicts(self, train_files: List[Dict], val_files: List[Dict], 
                                 batch_size: int = 2) -> Tuple[DataLoader, DataLoader]:
    """从给定的数据字典创建数据加载器"""
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
```
- 支持缓存数据集和普通数据集
- 灵活的批次大小配置
- 多进程数据加载

### train.py - 训练模块

这个文件实现了完整的模型训练流程，支持单模型和集成模型训练。

#### 主要类和函数详解：

**第1-25行：导入声明**
```python
import os
import time
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any
import matplotlib.pyplot as plt
from pathlib import Path

from DatasetLoader_transforms import DatasetLoader
from model import BasicModelBank, BankModelIntegration, create_optimizer, create_scheduler, create_full_ensemble
from utils import EarlyStopping, ModelCheckpoint, MetricsTracker
from monai.utils import set_determinism
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose
```
- 导入训练所需的所有模块
- 包括数据加载、模型、工具函数等

**第26-65行：ModelTrainer类初始化**
```python
class ModelTrainer:
    """BraTS脑肿瘤分割训练器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化训练器"""
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 设置随机种子
        set_determinism(seed=config.get('seed', 42))
        
        # 创建输出目录
        self.output_dir = Path(config.get('output_dir', './outputs'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self._setup_data()
        self._setup_model()
        self._setup_training()
        self._setup_logging()
```
- 初始化训练器配置
- 设置设备和随机种子
- 创建输出目录
- 初始化各个组件

**第66-85行：数据设置方法**
```python
def _setup_data(self):
    """设置数据加载器"""
    print("设置数据加载器...")
    
    data_loader = DatasetLoader(
        data_dir=self.config['data_dir'],
        cache_rate=self.config.get('cache_rate', 0.1),
        num_workers=self.config.get('num_workers', 4),
        seed=self.config.get('seed', 42)
    )
    
    self.train_loader, self.val_loader = data_loader.get_dataloaders(
        batch_size=self.config.get('batch_size', 2)
    )
```
- 创建数据加载器实例
- 配置缓存率、工作进程数等参数
- 获取训练和验证数据加载器

**第86-150行：模型设置方法**
```python
def _setup_model(self):
    """设置模型、损失函数和指标"""
    print("设置模型...")
    
    # 检查是否使用集成模型
    use_ensemble = self.config.get('use_ensemble', False)
    
    if use_ensemble:
        print("使用集成模型")
        self.ensemble_model = create_full_ensemble(device=self.device)
        self.is_ensemble = True
        
        # 为集成模型创建单独的损失函数和指标
        from monai.losses import DiceCELoss
        from monai.metrics import DiceMetric, HausdorffDistanceMetric
        
        self.loss_function = DiceCELoss(
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            jaccard=False,
            reduction="mean"
        )
```
- 支持单模型和集成模型两种模式
- 为集成模型创建专门的损失函数和指标
- 计算和显示模型参数信息

**第151-220行：训练组件设置**
```python
def _setup_training(self):
    """设置训练组件"""
    print("设置训练组件...")
    
    if self.is_ensemble:
        # 为集成模型中的每个子模型创建优化器
        self.optimizers = []
        self.schedulers = []
        
        for model in self.ensemble_model.models:
            optimizer = create_optimizer(
                model.model,
                optimizer_name=self.config.get('optimizer', 'adamw'),
                learning_rate=self.config.get('learning_rate', 1e-4),
                weight_decay=self.config.get('weight_decay', 1e-5)
            )
            self.optimizers.append(optimizer)
            
            scheduler = create_scheduler(
                optimizer,
                scheduler_name=self.config.get('scheduler', 'cosineannealinglr'),
                T_max=self.config.get('max_epochs', 500)
            )
            self.schedulers.append(scheduler)
```
- 为每个子模型创建独立的优化器和调度器
- 配置早停和模型检查点
- 初始化指标跟踪器

**第221-320行：训练epoch方法**
```python
def train_epoch(self, epoch: int) -> Dict[str, float]:
    """训练一个epoch"""
    if self.is_ensemble:
        # 集成模型训练
        for model in self.ensemble_model.models:
            model.model.train()
    else:
        self.model.train()
    
    epoch_loss = 0
    num_batches = len(self.train_loader)
    
    # 重置指标
    for metric in self.metrics.values():
        metric.reset()
    
    progress_bar = tqdm(self.train_loader, desc=f"训练 Epoch {epoch+1}")
    
    for batch_idx, batch_data in enumerate(progress_bar):
        inputs = batch_data['image'].to(self.device)
        labels = batch_data['label'].to(self.device)
        
        if self.is_ensemble:
            # 集成模型训练
            total_loss = 0
            ensemble_outputs = []
            
            # 为每个子模型计算损失并更新
            for i, (model, optimizer) in enumerate(zip(self.ensemble_model.models, self.optimizers)):
                optimizer.zero_grad()
                
                # 前向传播
                outputs = model.model(inputs)
                ensemble_outputs.append(outputs)
                
                # 计算损失
                loss = self.loss_function(outputs, labels)
                total_loss += loss.item()
                
                # 反向传播
                loss.backward()
                optimizer.step()
```
- 支持单模型和集成模型的训练
- 实现完整的前向传播、损失计算、反向传播流程
- 实时显示训练进度和指标

**第321-400行：验证epoch方法**
```python
def validate_epoch(self, epoch: int) -> Dict[str, float]:
    """验证一个epoch"""
    if self.is_ensemble:
        # 集成模型验证
        for model in self.ensemble_model.models:
            model.model.eval()
    else:
        self.model.eval()
    
    epoch_loss = 0
    num_batches = len(self.val_loader)
    
    # 重置指标
    for metric in self.metrics.values():
        metric.reset()
    
    progress_bar = tqdm(self.val_loader, desc=f"验证 Epoch {epoch+1}")
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(progress_bar):
            inputs = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            
            if self.is_ensemble:
                # 使用集成模型预测
                outputs = self.ensemble_model.predict(inputs)
            else:
                # 使用滑动窗口推理
                outputs = self.model_creator.sliding_window_inference(inputs)
```
- 验证模式下禁用梯度计算
- 使用滑动窗口推理提高准确性
- 计算验证损失和指标

**第401-530行：完整训练流程**
```python
def train(self):
    """执行完整的训练流程"""
    print("开始训练...")
    print(f"最大训练轮数: {self.config.get('max_epochs', 500)}")
    print(f"输出目录: {self.output_dir}")
    print("-" * 50)
    
    max_epochs = self.config.get('max_epochs', 500)
    best_metric = -1
    
    for epoch in range(max_epochs):
        start_time = time.time()
        
        # 训练
        train_metrics = self.train_epoch(epoch)
        
        # 验证
        val_metrics = self.validate_epoch(epoch)
        
        # 更新学习率
        if self.is_ensemble:
            for scheduler in self.schedulers:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_metrics['val_loss'])
                else:
                    scheduler.step()
        else:
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['val_loss'])
            else:
                self.scheduler.step()
        
        # 记录指标
        all_metrics = {**train_metrics, **val_metrics}
        self.metrics_tracker.update(all_metrics)
        
        # 记录到TensorBoard
        for key, value in all_metrics.items():
            self.writer.add_scalar(f'Metrics/{key}', value, epoch)
        
        # 保存最佳模型
        current_metric = val_metrics['val_dice']
        if current_metric > best_metric:
            best_metric = current_metric
            
            if self.is_ensemble:
                # 保存集成模型
                ensemble_state = {
                    'epoch': epoch,
                    'model_states': [model.model.state_dict() for model in self.ensemble_model.models],
                    'optimizer_states': [opt.state_dict() for opt in self.optimizers],
                    'scheduler_states': [sch.state_dict() for sch in self.schedulers],
                    'best_metric': best_metric,
                    'config': self.config,
                    'is_ensemble': True,
                    'model_names': [model.model_name for model in self.ensemble_model.models]
                }
                self.checkpoint.save(ensemble_state)
```
- 完整的训练循环实现
- 自动保存最佳模型
- 早停机制防止过拟合
- TensorBoard日志记录
- 训练历史保存和可视化

### evaluate.py - 评估模块

这个文件实现了训练好的模型的性能评估功能。

#### 主要类和函数详解：

**第1-30行：文档和导入**
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BraTS脑肿瘤分割模型评估脚本

这个脚本用于评估训练好的BraTS脑肿瘤分割模型的性能。
支持CPU和GPU(CUDA)设备，可通过命令行参数指定模型路径和数据集路径。

使用示例:
    python evaluate.py --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/dataset
    python evaluate.py --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/dataset --device cpu
    python evaluate.py --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/dataset --device cuda --output_dir ./my_results

作者: 个人使用版本
版本: 3.1.0
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
```
- 详细的脚本说明和使用示例
- 导入评估所需的所有模块

**第31-65行：BraTSEvaluator类初始化**
```python
class BraTSEvaluator:
    """BraTS脑肿瘤分割模型评估器"""
    
    def __init__(self, 
                 model_path: str,
                 data_dir: str,
                 device: str = "cuda",
                 output_dir: str = "./evaluation_results"):
        """初始化评估器"""
        self.model_path = model_path
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置随机种子
        set_determinism(seed=42)
        
        # 初始化组件
        self._load_model()
        self._setup_data()
        self._setup_metrics()
```
- 初始化评估器参数
- 自动创建输出目录
- 设置随机种子确保可重复性

**第66-150行：模型加载方法**
```python
def _load_model(self):
    """加载训练好的模型"""
    print(f"加载模型: {self.model_path}")
    
    # 加载检查点
    checkpoint = torch.load(self.model_path, map_location=self.device)
    config = checkpoint.get('config', {})
    
    # 获取模型名称
    model_name = config.get('model_name', 'UNet')
    
    # 检查是否为集成模型
    use_ensemble = config.get('use_ensemble', False)
    
    if use_ensemble:
        print("检测到集成模型，使用集成评估模式")
        # 获取集成模型中的模型列表
        # 获取所有支持的模型作为默认集成
        default_ensemble = get_all_supported_models()
        ensemble_models = config.get('ensemble_models', default_ensemble)
        print(f"集成模型包含: {ensemble_models}")
        
        # 创建集成模型
        models = []
        for model_name in ensemble_models:
            model_creator = BasicModelBank(model_name=model_name, device=self.device)
            models.append(model_creator)
        
        self.ensemble_model = BankModelIntegration(models, device=self.device)
        self.is_ensemble = True
        print(f"成功创建集成模型，包含 {len(ensemble_models)} 个子模型")
```
- 智能检测单模型和集成模型
- 自动加载相应的模型权重
- 显示详细的模型信息

**第151-200行：数据和指标设置**
```python
def _setup_data(self):
    """设置数据加载器"""
    print("设置数据加载器...")
    
    data_loader = DatasetLoader(
        data_dir=self.data_dir,
        cache_rate=0.0,  # 评估时不使用缓存
        num_workers=2,
        seed=42
    )
    
    # 获取验证数据
    _, self.val_loader = data_loader.get_dataloaders(batch_size=1)  # 评估时使用batch_size=1
    
    print(f"验证样本数: {len(self.val_loader)}")

def _setup_metrics(self):
    """设置评估指标"""
    # 分割指标
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
```
- 评估时不使用数据缓存
- 使用batch_size=1确保准确性
- 设置多种评估指标

**第201-320行：模型评估方法**
```python
def evaluate_model(self) -> Dict[str, float]:
    """评估模型性能"""
    print("开始模型评估...")
    
    # 重置指标
    self.dice_metric.reset()
    self.hd_metric.reset()
    self.surface_metric.reset()
    
    all_dice_scores = []
    all_hd_scores = []
    all_surface_scores = []
    
    case_results = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(self.val_loader, desc="评估进度")):
            inputs = batch_data['image'].to(self.device)
            labels = batch_data['label'].to(self.device)
            subject_id = batch_data.get('subject_id', [f'case_{batch_idx}'])[0]
            
            # 使用滑动窗口推理
            if self.is_ensemble:
                # 集成模型推理
                outputs = self.ensemble_model.predict(inputs)
            else:
                # 单个模型推理
                outputs = self.model_creator.sliding_window_inference(inputs)
            
            # 后处理
            outputs_list = decollate_batch(outputs)
            labels_list = decollate_batch(labels)
            
            outputs_convert = [self.post_pred(pred) for pred in outputs_list]
            labels_convert = [self.post_label(label) for label in labels_list]
            
            # 计算指标
            dice_scores = self.dice_metric(y_pred=outputs_convert, y=labels_convert)
            hd_scores = self.hd_metric(y_pred=outputs_convert, y=labels_convert)
            surface_scores = self.surface_metric(y_pred=outputs_convert, y=labels_convert)
```
- 逐个案例进行评估
- 使用滑动窗口推理提高准确性
- 计算多种评估指标
- 保存每个案例的详细结果

**第321-400行：可视化保存方法**
```python
def _save_visualization(self, 
                      images: np.ndarray,
                      labels: np.ndarray, 
                      predictions: np.ndarray,
                      subject_id: str,
                      dice_score: float):
    """保存可视化结果"""
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
```
- 生成直观的分割结果可视化
- 显示多模态图像和分割对比
- 自动保存高质量图像

**第401-498行：结果保存和分析**
```python
def _save_detailed_results(self, case_results: List[Dict], summary_results: Dict[str, float]):
    """保存详细评估结果"""
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
```
- 保存详细的案例级别结果
- 生成统计摘要报告
- 创建结果分布图和箱线图

### utils.py - 工具函数

这个文件提供了训练和评估过程中需要的各种工具类和函数。

#### 主要类和函数详解：

**第1-15行：导入声明**
```python
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Optional
import time
from pathlib import Path
```
- 导入工具函数所需的基础模块

**第16-50行：EarlyStopping类**
```python
class EarlyStopping:
    """早停机制，防止过拟合"""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001, mode: str = 'min'):
        """
        初始化早停机制
        
        Args:
            patience: 容忍轮数
            min_delta: 最小改善幅度
            mode: 监控模式 ('min' 或 'max')
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """检查是否应该早停"""
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
```
- 实现早停机制防止过拟合
- 支持最小化和最大化两种监控模式
- 可配置容忍轮数和最小改善幅度

**第51-100行：ModelCheckpoint类**
```python
class ModelCheckpoint:
    """模型检查点保存器"""
    
    def __init__(self, save_dir: str, filename: str = 'best_model.pth'):
        """
        初始化模型检查点保存器
        
        Args:
            save_dir: 保存目录
            filename: 文件名
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.filename = filename
        self.filepath = self.save_dir / filename
        
    def save(self, state_dict: Dict[str, Any]):
        """保存模型检查点"""
        torch.save(state_dict, self.filepath)
        
    def load(self) -> Dict[str, Any]:
        """加载模型检查点"""
        if self.filepath.exists():
            return torch.load(self.filepath)
        else:
            raise FileNotFoundError(f"检查点文件不存在: {self.filepath}")
```
- 自动创建保存目录
- 提供保存和加载检查点的方法
- 支持灵活的文件命名

**第101-180行：MetricsTracker类**
```python
class MetricsTracker:
    """指标跟踪器，用于记录训练过程中的各种指标"""
    
    def __init__(self):
        """初始化指标跟踪器"""
        self.history = {}
        
    def update(self, metrics: Dict[str, float]):
        """更新指标"""
        for key, value in metrics.items():
            if key not in self.history:
                self.history[key] = []
            self.history[key].append(value)
            
    def get_history(self) -> Dict[str, List[float]]:
        """获取历史记录"""
        return self.history
        
    def get_latest(self) -> Dict[str, float]:
        """获取最新指标"""
        latest = {}
        for key, values in self.history.items():
            if values:
                latest[key] = values[-1]
        return latest
        
    def get_best(self, metric_name: str, mode: str = 'max') -> float:
        """获取最佳指标值"""
        if metric_name not in self.history:
            return None
            
        values = self.history[metric_name]
        if not values:
            return None
            
        if mode == 'max':
            return max(values)
        else:
            return min(values)
```
- 记录训练过程中的所有指标
- 支持获取历史记录、最新值、最佳值
- 提供保存和加载历史记录的功能

**第181-250行：VisualizationUtils类**
```python
class VisualizationUtils:
    """可视化工具类"""
    
    @staticmethod
    def plot_training_metrics(history: Dict[str, List[float]], 
                            save_path: Optional[str] = None):
        """绘制训练指标曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('训练过程监控', fontsize=16)
        
        # 损失曲线
        if 'loss' in history and 'val_loss' in history:
            axes[0, 0].plot(history['loss'], label='训练损失')
            axes[0, 0].plot(history['val_loss'], label='验证损失')
            axes[0, 0].set_title('损失曲线')
            axes[0, 0].set_xlabel('Epoch')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True)
        
        # Dice指标曲线
        if 'dice' in history and 'val_dice' in history:
            axes[0, 1].plot(history['dice'], label='训练Dice')
            axes[0, 1].plot(history['val_dice'], label='验证Dice')
            axes[0, 1].set_title('Dice系数曲线')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Dice Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # 学习率曲线
        if 'learning_rate' in history:
            axes[1, 0].plot(history['learning_rate'], label='学习率')
            axes[1, 0].set_title('学习率变化')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Learning Rate')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Hausdorff距离曲线
        if 'hausdorff' in history and 'val_hausdorff' in history:
            axes[1, 1].plot(history['hausdorff'], label='训练HD')
            axes[1, 1].plot(history['val_hausdorff'], label='验证HD')
            axes[1, 1].set_title('Hausdorff距离曲线')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Hausdorff Distance')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"训练曲线已保存到: {save_path}")
        
        plt.show()

**第251-280行：时间格式化函数**
```python
def format_time(seconds: float) -> str:
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}秒"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}分钟"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}小时"
```
- 将秒数转换为易读的时间格式
- 自动选择合适的时间单位
- 用于显示训练和评估耗时

**第281-320行：测试代码**
```python
if __name__ == "__main__":
    print("=" * 50)
    print("🔧 工具类测试")
    print("=" * 50)
    
    # 测试早停机制
    print("\n1. 测试早停机制:")
    early_stopping = EarlyStopping(patience=3, min_delta=0.01, mode='min')
    test_losses = [1.0, 0.8, 0.7, 0.71, 0.72, 0.73]  # 模拟损失值
    
    for epoch, loss in enumerate(test_losses):
        should_stop = early_stopping(loss)
        print(f"  Epoch {epoch}: Loss={loss:.2f}, 早停={should_stop}")
        if should_stop:
            break
    
    # 测试指标跟踪器
    print("\n2. 测试指标跟踪器:")
    tracker = MetricsTracker()
    
    # 模拟训练过程
    for epoch in range(5):
        metrics = {
            'loss': 1.0 - epoch * 0.1,
            'dice': 0.5 + epoch * 0.1,
            'val_loss': 1.1 - epoch * 0.08,
            'val_dice': 0.45 + epoch * 0.08
        }
        tracker.update(metrics)
    
    print(f"  最新指标: {tracker.get_latest()}")
    print(f"  最佳Dice: {tracker.get_best('dice', 'max'):.3f}")
    
    # 测试时间格式化
    print("\n3. 测试时间格式化:")
    test_times = [30, 150, 3720, 7380]
    for t in test_times:
        print(f"  {t}秒 = {format_time(t)}")
    
    print("\n✅ 所有工具类已准备就绪!")
    print("\n📝 注意: 配置管理功能已集成到main.py中的get_high_performance_config函数")
```
- 完整的工具类功能测试
- 验证早停机制、指标跟踪器和时间格式化功能
- 提供使用示例和测试结果

## 使用方法

### 1. 数据准备

确保你的BraTS数据集按照以下结构组织：

```
BraTS_data/
├── BraTS-GLI-00000-000/
│   ├── BraTS-GLI-00000-000-t1n.nii.gz    # T1 native
│   ├── BraTS-GLI-00000-000-t1c.nii.gz    # T1 contrast-enhanced  
│   ├── BraTS-GLI-00000-000-t2w.nii.gz    # T2 weighted
│   ├── BraTS-GLI-00000-000-t2f.nii.gz    # T2 FLAIR
│   └── BraTS-GLI-00000-000-seg.nii.gz    # 分割标注（训练时必需）
├── BraTS-GLI-00001-000/
│   ├── BraTS-GLI-00001-000-t1n.nii.gz
│   ├── BraTS-GLI-00001-000-t1c.nii.gz
│   ├── BraTS-GLI-00001-000-t2w.nii.gz
│   ├── BraTS-GLI-00001-000-t2f.nii.gz
│   └── BraTS-GLI-00001-000-seg.nii.gz
└── ...
```

**重要提示**：
- 数据集会自动按8:2比例划分为训练集和验证集
- 文件命名必须严格按照上述格式
- 分割文件在训练时必需，评估时可选

### 2. 训练模型

#### 基础训练
```bash
# 使用默认UNet模型训练
python main.py --mode train --data_dir /path/to/BraTS_data

# 指定特定模型
python main.py --mode train --model_name SegResNet --data_dir /path/to/BraTS_data

# 自定义训练参数
python main.py --mode train \
    --model_name UNet \
    --epochs 200 \
    --batch_size 4 \
    --learning_rate 1e-4 \
    --data_dir /path/to/BraTS_data \
    --output_dir ./my_outputs
```

#### 多模型训练

项目支持两种多模型训练模式：

**使用示例**：

以下是6种主要的训练模式，展示了不同参数组合的使用方法：

**1. 多模型默认集成且并行训练**：
```bash
# 指定3个模型，默认parallel=true，自动启用集成训练
python main.py --mode train --data_dir /path/to/BraTS_data --model_names UNet SegResNet UNETR
```

**2. 多模型集成且并行训练（显式指定）**：
```bash
# 指定3个模型，显式设置parallel=true，自动启用集成训练
python main.py --mode train --data_dir /path/to/BraTS_data --model_names UNet SegResNet UNETR --parallel true
```

**3. 多模型集成且逐个训练**：
```bash
# 指定3个模型，设置parallel=false，自动启用集成训练，逐个训练
python main.py --mode train --data_dir /path/to/BraTS_data --model_names UNet SegResNet UNETR --parallel false
```

**4. 完整集成默认并行训练**：
```bash
# 启用ensemble，默认parallel=true，训练所有7个模型
python main.py --mode train --data_dir /path/to/BraTS_data --enable_ensemble
```

**5. 完整集成并行训练（显式指定）**：
```bash
# 启用ensemble，显式设置parallel=true，训练所有7个模型
python main.py --mode train --data_dir /path/to/BraTS_data --enable_ensemble --parallel true
```

**6. 完整集成逐个训练**：
```bash
# 启用ensemble，设置parallel=false，训练所有7个模型，逐个训练
python main.py --mode train --data_dir /path/to/BraTS_data --enable_ensemble --parallel false
```

**参数说明**：
- `--model_names`: 指定要训练的模型列表，当指定多个模型时自动启用集成训练
- `--enable_ensemble`: 启用完整集成训练，如果没有指定model_names则训练所有7个模型
- `--parallel`: 控制训练方式，true（默认）为并行训练，false为逐个训练

**集成机制**：
当指定多个模型时，系统会自动启用集成训练模式。集成是指将多个不同的模型架构训练后，通过平均预测和投票机制融合成一个新的集成模型，而不是简单地选择其中最好的一个模型。

**逐个集成训练（节省内存）**：
```bash
# 训练集成模型（逐个模式，使用BankModelIntegration）
python main.py --mode train \
    --enable_ensemble \
    --parallel false \
    --epochs 300 \
    --data_dir /path/to/BraTS_data

# 集成训练结合指定模型（逐个模式）
python main.py --mode train \
    --enable_ensemble \
    --model_names UNet SegResNet UNETR \
    --parallel false \
    --epochs 250 \
    --data_dir /path/to/BraTS_data
```

### 3. 模型评估

```bash
# 评估单个模型
python main.py --mode eval \
    --model_path ./outputs/checkpoints/best_model.pth \
    --data_dir /path/to/BraTS_data

# 评估集成模型
python main.py --mode eval \
    --model_path ./outputs/checkpoints/best_model.pth \
    --data_dir /path/to/BraTS_data

# 自定义输出目录
python main.py --mode eval \
    --model_path ./outputs/checkpoints/best_model.pth \
    --data_dir /path/to/BraTS_data \
    --output_dir ./my_evaluation_results

# 指定设备评估
python main.py --mode eval \
    --model_path ./outputs/checkpoints/best_model.pth \
    --data_dir /path/to/BraTS_data \
    --device cpu

# 使用自动设备检测
python main.py --mode eval \
    --model_path ./outputs/checkpoints/best_model.pth \
    --data_dir /path/to/BraTS_data \
    --device auto
```

### 4. 支持的模型架构

项目支持以下深度学习模型：

- **UNet**: 经典的U型网络，适合医学图像分割
- **SegResNet**: 基于ResNet的分割网络
- **UNETR**: 基于Transformer的U型网络
- **SwinUNETR**: 基于Swin Transformer的分割网络
- **AttentionUNet**: 带注意力机制的U型网络
- **VNet**: 专为3D医学图像设计的网络
- **HighResNet**: 高分辨率网络

### 5. 输出文件说明

训练完成后，会在`./outputs`目录下生成以下文件：

```
outputs/
├── checkpoints/
│   ├── best_model.pth          # 最佳模型权重（单模型或集成模型）
│   ├── model_20240101_120000.pth # 带时间戳的备份文件

├── logs/
│   └── tensorboard_logs/       # TensorBoard日志
├── metrics/
│   ├── training_history.json   # 训练历史记录
│   └── training_curves.png     # 训练曲线图
└── visualizations/
    └── sample_predictions.png  # 样本预测可视化
```

#### 集成模型保存详情

**最佳集成模型保存位置**：`./outputs/checkpoints/best_model.pth`

集成模型文件包含以下内容：
- `model_states`：所有子模型的状态字典列表
- `optimizer_states`：所有优化器状态列表
- `scheduler_states`：所有调度器状态列表  
- `best_metric`：最佳验证指标（Dice分数）
- `config`：完整的训练配置信息
- `is_ensemble`：标识为集成模型（True）
- `model_names`：包含的模型名称列表
- `save_time`：保存时间戳
- `epoch`：保存时的训练轮数

**备份机制**：
- 系统会自动创建带时间戳的备份文件
- 格式：`model_YYYYMMDD_HHMMSS.pth`
- 位置：同样在`./outputs/checkpoints/`目录下

**自定义保存路径**：
```bash
# 自定义输出目录
python main.py --mode train --enable_ensemble --output_dir ./my_custom_output
# 集成模型将保存在：./my_custom_output/checkpoints/best_model.pth
```

**评估集成模型**：
```bash
python main.py --mode eval --model_path ./outputs/checkpoints/best_model.pth --data_dir /path/to/data
```

评估完成后，会在指定的输出目录下生成：

```
evaluation_results/
├── case_results.csv           # 每个案例的详细结果
├── summary_results.txt        # 总体统计结果
├── results_distribution.png   # 结果分布图
└── visualizations/
    ├── case_001_prediction.png
    ├── case_002_prediction.png
    └── ...
```

### 6. 性能优化建议

#### GPU内存优化
- 如果遇到GPU内存不足，可以减小`batch_size`
- 使用`--auto_adjust`参数自动调整参数
- 降低`cache_rate`减少内存占用

#### 训练速度优化
- 使用更多的`num_workers`加速数据加载
- 启用混合精度训练（AMP）
- 使用SSD存储数据集

#### 模型性能优化
- 使用集成模型提高准确性
- 调整学习率和优化器参数
- 使用数据增强提高泛化能力

### 7. 常见问题解决

#### 数据加载问题
```bash
# 检查数据目录结构
python -c "from DatasetLoader_transforms import DatasetLoader; loader = DatasetLoader('/path/to/data'); print('数据检查完成')"
```

#### 模型加载问题
```bash
# 检查模型文件
python -c "import torch; checkpoint = torch.load('model.pth', map_location='cpu'); print('模型文件正常')"
```

#### 设备配置问题
```bash
# 检查CUDA可用性
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU数量: {torch.cuda.device_count()}')"
```

## 项目特点

1. **模块化设计**: 清晰的代码结构，易于维护和扩展
2. **多模型支持**: 支持7种不同的深度学习架构
3. **集成学习**: 支持多模型集成提高性能
4. **自动优化**: 根据硬件自动调整参数
5. **完整评估**: 多种评估指标和可视化
6. **易于使用**: 简单的命令行接口
7. **高性能**: 优化的数据加载和训练流程

## 技术栈

- **深度学习框架**: PyTorch + MONAI
- **数据处理**: NumPy + NiBabel
- **可视化**: Matplotlib + TensorBoard
- **数据分析**: Pandas
- **进度显示**: tqdm

## 模型部署

训练完成后，可以将集成模型部署到生产环境中进行实际应用。项目提供了完整的部署解决方案。

### 快速部署

#### 单文件推理
```bash
# 对单个医学图像进行分割预测
python deploy.py \
    --model_path ./outputs/checkpoints/best_model.pth \
    --input_file /path/to/input.nii.gz \
    --output_file /path/to/output.nii.gz
```

#### 批量推理
```bash
# 批量处理多个文件
python deploy.py \
    --model_path ./outputs/checkpoints/best_model.pth \
    --input_dir /path/to/input_directory \
    --output_dir /path/to/output_directory
```

#### API服务部署
```bash
# 启动REST API服务
python deploy.py \
    --model_path ./outputs/checkpoints/best_model.pth \
    --api_mode \
    --port 8080

# 使用API进行预测
curl -X POST \
  -F "file=@input_image.nii.gz" \
  http://localhost:8080/predict \
  -o prediction_result.nii.gz
```

### Docker容器化部署

```bash
# 构建Docker镜像
docker build -t brats-model:latest .

# 运行容器
docker run -d \
    --name brats-api \
    --gpus all \
    -p 8080:8080 \
    -v $(pwd)/outputs/checkpoints:/app/models:ro \
    brats-model:latest

# 或使用Docker Compose
docker-compose up -d
```

### 部署特性

- **多种部署方式**: 本地部署、Docker容器化、Kubernetes集群
- **REST API接口**: 标准HTTP接口，易于集成
- **自动设备检测**: 自动选择CPU/GPU进行推理
- **批量处理**: 支持大规模数据处理
- **健康检查**: 内置服务监控和健康检查
- **日志记录**: 完整的推理日志和错误追踪
- **性能优化**: 滑动窗口推理和内存优化

### API接口

| 接口 | 方法 | 描述 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/info` | GET | 模型信息 |
| `/predict` | POST | 图像分割预测 |

### 详细部署指南

完整的部署文档请参考 [DEPLOYMENT.md](DEPLOYMENT.md)，包含：

- 环境要求和配置
- 多种部署方式详解
- 性能优化建议
- 监控和维护指南
- 故障排除方案
- 安全配置建议

