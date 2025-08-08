# 基于MONAI框架的医学图像分割解决方案

## 项目概述

本项目是一个基于MONAI框架的医学图像分割解决方案，支持多种深度学习模型架构和训练策略。项目提供了从基础模型训练到高级技术（知识蒸馏、融合网络、神经架构搜索）的完整解决方案，并包含全面的评估指标体系。

### 核心特性

- **多模型支持**: 8种基础模型架构（UNet、SegResNet、UNETR、SwinUNETR、AttentionUNet、VNet、HighResNet、VNet3D）
- **高级训练策略**: 知识蒸馏、融合网络、神经架构搜索（NAS）、NAS-蒸馏集成
- **多数据集支持**: BraTS2024、MS_MultiSpine等医学图像分割数据集
- **完整评估体系**: 7种评估指标（Dice、Hausdorff距离、表面距离、混淆矩阵、IoU、广义Dice、FROC）
- **自适应损失函数**: 5阶段动态权重调整策略
- **智能模型管理**: 自动避免教师-学生模型重复，智能参数调整

### 支持的数据集

| 数据集 | 输入通道 | 输出类别 | 描述 |
|--------|----------|----------|------|
| BraTS2024 | 4 | 4 | T1, T1ce, T2, FLAIR → ET, TC, WT, Background |
| MS_MultiSpine | 2 | 6 | T2, STIR/PSIR/MP2RAGE → 6种病变类别 |

## 环境配置

### 系统要求

- Python 3.8+
- CUDA 11.0+ (推荐)
- 内存: 16GB+ (推荐32GB+)
- 显存: 8GB+ (推荐16GB+)

### 依赖安装

```bash
# 安装基础依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install monai[all]
pip install matplotlib pandas scikit-learn tqdm
pip install scipy numpy pillow

# 安装可选依赖（用于高级功能）
pip install tensorboard wandb
```

### 数据准备

```bash
# 创建数据目录
mkdir -p ./data/BraTS2024
mkdir -p ./data/MS_MultiSpine

# 下载并解压数据集到对应目录
# BraTS2024: 将数据放置在 ./data/BraTS2024/
# MS_MultiSpine: 将数据放置在 ./data/MS_MultiSpine/
```

## 参数详细说明

### 通用参数

以下参数适用于所有训练、评估、推理模式：

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--mode` | str | None | 是 | 运行模式：train/evaluate/inference |
| `--data_dir` | str | None | 是 | 数据集根目录路径 |
| `--dataset_type` | str | auto | 否 | 数据集类型：BraTS/MS_MultiSpine/auto |
| `--output_dir` | str | ./outputs | 否 | 输出目录路径 |
| `--device` | str | auto | 否 | 计算设备：cpu/cuda/auto |
| `--batch_size` | int | None | 否 | 批次大小（覆盖配置文件设置） |
| `--epochs` | int | 500 | 否 | 训练轮数 |
| `--learning_rate` | float | None | 否 | 学习率（覆盖配置文件设置） |
| `--auto_adjust` | bool | True | 否 | 是否根据设备自动调节参数 |

### 模型选择参数

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--model_category` | str | None | 是 | 模型类别：basic/advanced |
| `--model_name` | str | None | 条件 | 单个基础模型名称 |
| `--model_names` | list | None | 条件 | 多个基础模型名称列表 |
| `--model_type` | str | fusion | 条件 | 复合架构类型：fusion/distillation/nas/nas_distillation |
| `--parallel` | bool | True | 否 | 是否启用并行训练 |

### 知识蒸馏参数

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--teacher_models` | list | auto | 否 | 教师模型列表（默认使用所有基础模型） |
| `--student_model` | str | VNet3D | 否 | 学生模型名称 |
| `--distillation_temperature` | float | 4.0 | 否 | 蒸馏温度参数 |
| `--distillation_alpha` | float | 0.7 | 否 | 蒸馏损失权重 |
| `--pretrained_dir` | str | ./pretrained_teachers | 否 | 预训练教师模型目录 |
| `--pretrain_teachers` | bool | True | 是 | 启用教师模型预训练（默认启用） |
| `--teacher_epochs` | int | 100 | 否 | 教师模型预训练轮数 |
| `--force_retrain_teachers` | bool | False | 否 | 是否强制重新训练教师模型 |

### 融合网络参数

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--fusion_models` | list | auto | 否 | 融合网络基础模型列表 |
| `--fusion_type` | str | cross_attention | 否 | 融合类型：cross_attention/channel_attention/spatial_attention/adaptive |
| `--fusion_channels` | list | [64,128,256,512] | 否 | 融合网络通道配置 |

### NAS搜索参数

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--nas_epochs` | int | 50 | 否 | NAS搜索轮数 |
| `--nas_type` | str | supernet | 否 | NAS类型：supernet/searcher/progressive |
| `--base_channels` | int | 32 | 否 | NAS网络基础通道数 |
| `--num_layers` | int | 4 | 否 | NAS网络层数 |
| `--arch_lr` | float | 3e-4 | 否 | 架构参数学习率（DARTS） |
| `--model_lr` | float | 1e-3 | 否 | 模型权重学习率（DARTS） |
| `--max_layers` | int | 8 | 否 | 最大网络层数（渐进式NAS） |
| `--start_layers` | int | 2 | 否 | 起始网络层数（渐进式NAS） |

### NAS-蒸馏集成参数

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--distillation_epochs` | int | 100 | 否 | NAS搜索和教师预训练完成后的最终知识蒸馏训练轮数 |
| `--distillation_lr` | float | 1e-4 | 否 | 蒸馏阶段学习率 |
| `--nas_distillation_save_dir` | str | ./checkpoints/nas_distillation | 否 | NAS-蒸馏模型保存目录 |

### 评估参数

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--model_path` | str | None | 是 | 模型检查点路径 |

### 推理参数

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--input` | str | None | 是 | 输入文件或目录路径 |
| `--output` | str | None | 是 | 输出文件或目录路径 |
| `--batch_inference` | bool | False | 否 | 是否启用批量推理 |
| `--roi_size` | list | [128,128,128] | 否 | 滑动窗口大小 |
| `--sw_batch_size` | int | 4 | 否 | 滑动窗口批次大小 |
| `--overlap` | float | 0.6 | 否 | 滑动窗口重叠率 |
| `--no_visualization` | bool | False | 否 | 不保存可视化结果 |

## 文件存储位置说明

### 目录结构

```
项目根目录/
├── data/                           # 数据集目录
│   ├── BraTS2024/                 # BraTS数据集
│   └── MS_MultiSpine/             # MS_MultiSpine数据集
├── outputs/                        # 输出根目录
│   ├── models/                    # 模型保存目录
│   │   ├── basic_model/          # 基础模型
│   │   │   ├── UNet/             # 具体模型目录
│   │   │   │   ├── checkpoints/  # 检查点文件
│   │   │   │   │   ├── best_model.pth      # 最佳模型
│   │   │   │   │   ├── latest_model.pth    # 最新模型
│   │   │   │   │   └── epoch_*.pth         # 各轮次模型
│   │   │   │   ├── logs/         # 训练日志
│   │   │   │   └── config.json   # 模型配置
│   │   │   └── [其他模型]/
│   │   ├── fusion_model/         # 融合网络模型
│   │   ├── distillation_student/ # 知识蒸馏学生模型
│   │   ├── nas_model/            # NAS搜索模型
│   │   └── nas_distillation_student/ # NAS-蒸馏学生模型
│   ├── evaluation/               # 评估结果目录
│   │   ├── case_results.csv     # 案例级别结果
│   │   ├── summary_results.txt  # 总体统计结果
│   │   ├── results_distribution.png # 结果分布图
│   │   └── visualizations/      # 可视化图表目录
│   │       ├── all_metrics_distribution.png    # 所有指标分布图
│   │       ├── metrics_comparison.png          # 指标对比图
│   │       ├── froc_curve.png                 # FROC曲线
│   │       ├── confusion_matrix_heatmap.png   # 混淆矩阵热力图
│   │       └── metrics_correlation.png        # 指标相关性图
│   └── inference/               # 推理结果目录
│       ├── predictions/         # 预测结果
│       └── visualizations/      # 可视化结果
├── pretrained_teachers/         # 预训练教师模型目录
│   ├── UNet_pretrained.pth
│   ├── SegResNet_pretrained.pth
│   └── [其他教师模型].pth
└── checkpoints/                # 临时检查点目录
    └── nas_distillation/       # NAS-蒸馏临时文件
```

### 7个评估指标图片保存位置

所有评估指标的可视化图表保存在 `./outputs/evaluation/visualizations/` 目录下：

1. **all_metrics_distribution.png** - 所有7个指标的分布图
2. **metrics_comparison.png** - 指标对比分析图
3. **froc_curve.png** - FROC曲线图
4. **confusion_matrix_heatmap.png** - 混淆矩阵热力图
5. **metrics_correlation.png** - 指标相关性分析图
6. **results_distribution.png** - 基础结果分布图
7. **model_comparison.png** - 多模型对比图（如果进行模型比较）

### 文件命名规则

- **模型文件**: `{model_name}_{timestamp}.pth`
- **最佳模型**: `best_model.pth`
- **配置文件**: `config.json`
- **日志文件**: `training_log_{timestamp}.txt`
- **评估结果**: `case_results.csv`, `summary_results.txt`

## 训练指令详细说明

### 基础模型训练

基础模型训练使用单个或多个基础模型架构进行训练。

#### 基础训练参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_category` | str | basic | 模型类别，基础训练固定为basic |
| `--model_name` | str | None | 单个模型训练时使用 |
| `--model_names` | list | None | 多个模型训练时使用 |
| `--data_dir` | str | ./data | 数据集目录 |
| `--dataset_type` | str | BraTS | 数据集类型 |
| `--batch_size` | int | 2 | 批次大小 |
| `--epochs` | int | 100 | 训练轮数 |
| `--learning_rate` | float | 1e-4 | 学习率 |
| `--output_dir` | str | ./outputs | 输出目录 |

#### 基础训练示例

**最简单的训练指令**（使用默认参数）：
```bash
# 训练单个UNet模型（默认配置）
python main.py --mode train --model_category basic --model_name UNet
```

**中等复杂度训练指令**：
```bash
# 训练VNet3D模型，自定义数据集和基本参数
python main.py --mode train \
    --model_category basic \
    --model_name VNet3D \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS \
    --batch_size 4 \
    --epochs 150 \
    --learning_rate 2e-4
```

**完整参数训练指令**：
```bash
# 训练SegResNet模型，完整参数配置
python main.py --mode train \
    --model_category basic \
    --model_name SegResNet \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS \
    --output_dir ./outputs \
    --batch_size 4 \
    --epochs 200 \
    --learning_rate 1e-4 \
    --device cuda \
    --parallel \
    --auto_adjust
```

**多模型并行训练**：
```bash
# 同时训练多个基础模型
python main.py --mode train \
    --model_category basic \
    --model_names UNet SegResNet VNet3D UNETR \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS \
    --batch_size 2 \
    --epochs 100 \
    --parallel
```

### 知识蒸馏训练

知识蒸馏使用多个预训练的教师模型指导学生模型学习。

#### 知识蒸馏参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_category` | str | advanced | 模型类别，蒸馏训练固定为advanced |
| `--model_type` | str | distillation | 复合架构类型，蒸馏训练固定为distillation |
| `--teacher_models` | list | auto | 教师模型列表，默认使用所有基础模型 |
| `--student_model` | str | VNet3D | 学生模型名称 |
| `--distillation_temperature` | float | 4.0 | 蒸馏温度参数 |
| `--distillation_alpha` | float | 0.7 | 蒸馏损失权重 |
| `--pretrained_dir` | str | ./pretrained_teachers | 预训练教师模型目录 |
| `--pretrain_teachers` | bool | True | 启用教师模型预训练（默认启用） |
| `--teacher_epochs` | int | 50 | 教师模型预训练轮数 |

#### 知识蒸馏示例

**最简单的蒸馏指令**（使用默认配置）：
```bash
# 基础知识蒸馏，使用所有8个模型作为教师，VNet3D作为学生
python main.py --mode train \
    --model_category advanced \
    --model_type distillation
```

**中等复杂度蒸馏指令**：
```bash
# 自定义教师和学生模型的知识蒸馏
python main.py --mode train \
    --model_category advanced \
    --model_type distillation \
    --teacher_models UNet SegResNet UNETR SwinUNETR \
    --student_model AttentionUNet \
    --distillation_temperature 3.0 \
    --distillation_alpha 0.8 \
    --epochs 120
```

**完整参数蒸馏指令**：
```bash
# 完整配置的知识蒸馏训练
python main.py --mode train \
    --model_category advanced \
    --model_type distillation \
    --teacher_models UNet SegResNet UNETR SwinUNETR VNet HighResNet \
    --student_model VNet3D \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS \
    --output_dir ./outputs \
    --batch_size 3 \
    --epochs 150 \
    --learning_rate 1e-4 \
    --distillation_temperature 4.0 \
    --distillation_alpha 0.7 \
    --pretrained_dir ./pretrained_teachers \
    --teacher_epochs 50 \
    --device cuda
```

**禁用预训练的蒸馏**：
```bash
# 禁用教师模型预训练，直接使用随机初始化的教师模型
python main.py --mode train \
    --model_category advanced \
    --model_type distillation \
    --teacher_models UNet SegResNet VNet \
    --student_model VNet3D \
    --pretrain_teachers False \
    --epochs 120
```

**强制重新预训练教师模型**：
```bash
# 强制重新训练已存在的预训练教师模型
python main.py --mode train \
    --model_category advanced \
    --model_type distillation \
    --teacher_models UNet SegResNet VNet \
    --student_model VNet3D \
    --teacher_epochs 80 \
    --force_retrain_teachers \
    --epochs 120
```

### 融合网络训练

融合网络将多个基础模型的特征进行融合，形成更强的模型。

#### 融合网络参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_category` | str | advanced | 模型类别，融合训练固定为advanced |
| `--model_type` | str | fusion | 复合架构类型，融合训练固定为fusion |
| `--fusion_models` | list | auto | 融合网络基础模型列表 |
| `--fusion_type` | str | cross_attention | 融合类型 |
| `--fusion_channels` | list | [64,128,256,512] | 融合网络通道配置 |

#### 融合网络示例

**最简单的融合指令**（使用默认配置）：
```bash
# 基础融合网络，使用所有8个模型进行融合
python main.py --mode train \
    --model_category advanced \
    --model_type fusion
```

**中等复杂度融合指令**：
```bash
# 自定义融合模型和融合类型
python main.py --mode train \
    --model_category advanced \
    --model_type fusion \
    --fusion_models UNet SegResNet VNet3D UNETR \
    --fusion_type channel_attention \
    --epochs 100
```

**完整参数融合指令**：
```bash
# 完整配置的融合网络训练
python main.py --mode train \
    --model_category advanced \
    --model_type fusion \
    --fusion_models UNet SegResNet VNet3D UNETR SwinUNETR \
    --fusion_type cross_attention \
    --fusion_channels 32 64 128 256 512 \
    --data_dir ./data/MS_MultiSpine \
    --dataset_type MS_MultiSpine \
    --output_dir ./outputs \
    --batch_size 2 \
    --epochs 150 \
    --learning_rate 5e-5 \
    --device cuda
```

### NAS搜索训练

神经架构搜索自动寻找最优的网络架构。

#### NAS搜索参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_category` | str | advanced | 模型类别，NAS训练固定为advanced |
| `--model_type` | str | nas | 复合架构类型，NAS训练固定为nas |
| `--nas_epochs` | int | 50 | NAS搜索轮数 |
| `--nas_type` | str | supernet | NAS类型 |
| `--base_channels` | int | 32 | NAS网络基础通道数 |
| `--num_layers` | int | 4 | NAS网络层数 |
| `--arch_lr` | float | 3e-4 | 架构参数学习率 |
| `--model_lr` | float | 1e-3 | 模型权重学习率 |

#### NAS搜索示例

**最简单的NAS指令**（超网络搜索）：
```bash
# 基础NAS搜索，使用超网络方法
python main.py --mode train \
    --model_category advanced \
    --model_type nas
```

**DARTS搜索指令**：
```bash
# 使用DARTS可微分架构搜索
python main.py --mode train \
    --model_category advanced \
    --model_type nas \
    --nas_type searcher \
    --nas_epochs 80 \
    --arch_lr 3e-4 \
    --model_lr 1e-3
```

**渐进式NAS搜索指令**：
```bash
# 使用渐进式NAS搜索
python main.py --mode train \
    --model_category advanced \
    --model_type nas \
    --nas_type progressive \
    --max_layers 10 \
    --start_layers 3 \
    --nas_epochs 60
```

**完整参数NAS指令**：
```bash
# 完整配置的NAS搜索
python main.py --mode train \
    --model_category advanced \
    --model_type nas \
    --nas_type supernet \
    --nas_epochs 100 \
    --base_channels 64 \
    --num_layers 6 \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS \
    --output_dir ./outputs \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --device cuda
```

### NAS-蒸馏集成训练

NAS-蒸馏集成结合了神经架构搜索和知识蒸馏，先搜索最优架构，再用多教师蒸馏训练。

#### NAS-蒸馏参数说明

| 参数 | 类型 | 默认值 | 描述 |
|------|------|--------|------|
| `--model_category` | str | advanced | 模型类别，固定为advanced |
| `--model_type` | str | nas_distillation | 复合架构类型，固定为nas_distillation |
| `--teacher_models` | list | auto（默认使用所有8个基础模型：UNet、SegResNet、UNETR、SwinUNETR、AttentionUNet、VNet、HighResNet、VNet3D） | 教师模型列表 |
| `--pretrain_teachers` | bool | True | 启用教师模型预训练（默认启用） |
| `--teacher_epochs` | int | 50 | 教师模型预训练轮数 |
| `--pretrained_dir` | str | ./pretrained_teachers | 预训练模型保存目录 |
| `--nas_epochs` | int | 50 | NAS搜索轮数 |
| `--distillation_epochs` | int | 100 | NAS搜索和教师预训练完成后的最终知识蒸馏训练轮数 |
| `--distillation_lr` | float | 1e-4 | 蒸馏阶段学习率 |
| `--distillation_temperature` | float | 4.0 | 蒸馏温度参数 |
| `--distillation_alpha` | float | 0.7 | 蒸馏损失权重 |
| `--nas_distillation_save_dir` | str | ./checkpoints/nas_distillation | 保存目录 |

#### NAS-蒸馏示例

**最简单的NAS-蒸馏指令**：
```bash
# 基础NAS-蒸馏集成，使用所有8个模型作为教师
python main.py --mode train \
    --model_category advanced \
    --model_type nas_distillation
```

**中等复杂度NAS-蒸馏指令**：
```bash
# 自定义教师模型和训练轮数，启用教师预训练
python main.py --mode train \
    --model_category advanced \
    --model_type nas_distillation \
    --teacher_models UNet SegResNet UNETR VNet3D \
    --teacher_epochs 60 \
    --nas_epochs 60 \
    --distillation_epochs 120
```

**完整参数NAS-蒸馏指令**：
```bash
# 完整配置的NAS-蒸馏集成训练（启用教师预训练）
python main.py --mode train \
    --model_category advanced \
    --model_type nas_distillation \
    --teacher_models UNet SegResNet UNETR SwinUNETR VNet HighResNet \
    --teacher_epochs 80 \
    --pretrained_dir ./custom_pretrained_teachers \
    --nas_epochs 80 \
    --distillation_epochs 150 \
    --arch_lr 3e-4 \
    --model_lr 1e-3 \
    --distillation_lr 5e-5 \
    --distillation_temperature 4.0 \
    --distillation_alpha 0.8 \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS \
    --output_dir ./outputs \
    --batch_size 3 \
    --device cuda \
    --nas_distillation_save_dir ./checkpoints/nas_distillation_custom
```

**禁用教师预训练的NAS-蒸馏指令**：
```bash
# 禁用教师模型预训练的NAS-蒸馏集成训练
python main.py --mode train \
    --model_category advanced \
    --model_type nas_distillation \
    --teacher_models UNet SegResNet UNETR VNet3D \
    --pretrain_teachers False \
    --nas_epochs 60 \
    --distillation_epochs 120 \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS
```

## 评估指令详细说明

模型评估使用7种评估指标对训练好的模型进行全面评估。

### 评估参数说明

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--mode` | str | evaluate | 是 | 运行模式，评估固定为evaluate |
| `--model_path` | str | None | 是 | 模型检查点路径或目录 |
| `--data_dir` | str | ./data | 否 | 测试数据集目录 |
| `--dataset_type` | str | BraTS | 否 | 数据集类型 |
| `--output_dir` | str | ./outputs | 否 | 评估结果输出目录 |
| `--batch_size` | int | 1 | 否 | 评估批次大小 |
| `--device` | str | auto | 否 | 计算设备 |

### 评估示例

**最简单的评估指令**：
```bash
# 评估基础模型（自动查找best_model.pth）
python main.py --mode evaluate \
    --model_path ./outputs/models/basic_model/UNet/checkpoints/
```

**指定具体模型文件的评估**：
```bash
# 评估指定的模型文件
python main.py --mode evaluate \
    --model_path ./outputs/models/basic_model/VNet3D/checkpoints/best_model.pth \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS
```

**完整参数评估指令**：
```bash
# 完整配置的模型评估
python main.py --mode evaluate \
    --model_path ./outputs/models/distillation_student/checkpoints/best_model.pth \
    --data_dir ./data/BraTS2024 \
    --dataset_type BraTS \
    --output_dir ./outputs/evaluation \
    --batch_size 2 \
    --device cuda
```

**评估不同类型的模型**：
```bash
# 评估融合网络模型
python main.py --mode evaluate \
    --model_path ./outputs/models/fusion_model/checkpoints/best_model.pth

# 评估NAS搜索模型
python main.py --mode evaluate \
    --model_path ./outputs/models/nas_model/checkpoints/best_model.pth

# 评估NAS-蒸馏模型
python main.py --mode evaluate \
    --model_path ./outputs/models/nas_distillation_student/checkpoints/best_model.pth
```

### 评估输出说明

评估完成后，会在输出目录生成以下文件：

1. **case_results.csv** - 每个案例的详细评估结果
2. **summary_results.txt** - 总体统计结果
3. **visualizations/** - 7种评估指标的可视化图表
   - all_metrics_distribution.png
   - metrics_comparison.png
   - froc_curve.png
   - confusion_matrix_heatmap.png
   - metrics_correlation.png
   - results_distribution.png

## 推理指令详细说明

模型推理对新的医学图像进行分割预测。

### 推理参数说明

| 参数 | 类型 | 默认值 | 必需 | 描述 |
|------|------|--------|------|------|
| `--mode` | str | inference | 是 | 运行模式，推理固定为inference |
| `--model_path` | str | None | 是 | 模型检查点路径 |
| `--input` | str | None | 是 | 输入文件或目录路径 |
| `--output` | str | None | 是 | 输出文件或目录路径 |
| `--dataset_type` | str | BraTS | 否 | 数据集类型（影响输入通道数） |
| `--batch_inference` | bool | False | 否 | 是否启用批量推理 |
| `--roi_size` | list | [128,128,128] | 否 | 滑动窗口大小 |
| `--sw_batch_size` | int | 4 | 否 | 滑动窗口批次大小 |
| `--overlap` | float | 0.6 | 否 | 滑动窗口重叠率 |
| `--no_visualization` | bool | False | 否 | 不保存可视化结果 |
| `--device` | str | auto | 否 | 计算设备 |

### 推理示例

**最简单的推理指令**（单文件推理）：
```bash
# 对单个文件进行推理
python main.py --mode inference \
    --model_path ./outputs/models/basic_model/UNet/checkpoints/best_model.pth \
    --input ./data/test_case.nii.gz \
    --output ./results/prediction.nii.gz
```

**批量推理指令**：
```bash
# 对目录中的所有文件进行批量推理
python main.py --mode inference \
    --model_path ./outputs/models/distillation_student/checkpoints/best_model.pth \
    --input ./data/test_images/ \
    --output ./results/predictions/ \
    --batch_inference
```

**高质量推理指令**：
```bash
# 使用更大的滑动窗口和更高重叠率进行高质量推理
python main.py --mode inference \
    --model_path ./outputs/models/fusion_model/checkpoints/best_model.pth \
    --input ./data/test_images/ \
    --output ./results/high_quality_predictions/ \
    --roi_size 160 160 160 \
    --sw_batch_size 2 \
    --overlap 0.8 \
    --batch_inference
```

**完整参数推理指令**：
```bash
# 完整配置的模型推理
python main.py --mode inference \
    --model_path ./outputs/models/nas_distillation_student/checkpoints/best_model.pth \
    --input ./data/test_images/ \
    --output ./results/nas_distillation_predictions/ \
    --dataset_type BraTS \
    --batch_inference \
    --roi_size 128 128 128 \
    --sw_batch_size 4 \
    --overlap 0.6 \
    --device cuda
```

**不保存可视化的推理**：
```bash
# 只保存预测结果，不生成可视化图像（节省时间和空间）
python main.py --mode inference \
    --model_path ./outputs/models/basic_model/VNet3D/checkpoints/best_model.pth \
    --input ./data/test_images/ \
    --output ./results/predictions_only/ \
    --batch_inference \
    --no_visualization
```

### 推理输出说明

推理完成后，会在输出目录生成：

1. **预测结果文件** - .nii.gz格式的分割结果
2. **可视化图像** - .png格式的分割可视化（除非使用--no_visualization）
3. **推理日志** - 推理过程的详细日志

## 高级功能说明

### 自适应损失函数策略

项目实现了5阶段动态权重调整的自适应损失函数：

- **阶段1 (0-20%)**: 主要使用DiceCE损失 (70%)
- **阶段2 (20-40%)**: 增加Focal损失权重 (30%)
- **阶段3 (40-60%)**: 平衡各种损失函数
- **阶段4 (60-80%)**: 增加Tversky损失权重 (40%)
- **阶段5 (80-100%)**: 组合所有损失函数 (各20%)

### 智能重复检测

系统自动检测并避免教师-学生模型重复：

- 知识蒸馏时自动移除教师模型列表中的学生模型
- 教师模型列表为空时自动补充所有基础模型
- 提供详细的配置信息和警告提示

### 完整评估指标体系

项目支持7种评估指标：

1. **Dice系数** - 分割重叠度量
2. **Hausdorff距离** - 边界距离度量
3. **表面距离** - 表面相似性度量
4. **混淆矩阵** - 分类性能度量
5. **IoU (交并比)** - 区域重叠度量
6. **广义Dice** - 加权Dice度量
7. **FROC** - 检测性能度量

## 常见问题解答

### Q: 如何选择合适的模型？

A: 
- **计算资源充足**: 推荐UNETR或SwinUNETR（Transformer架构）
- **计算资源有限**: 推荐UNet或VNet3D（轻量级架构）
- **追求最佳性能**: 推荐融合网络或NAS-蒸馏集成
- **快速原型验证**: 推荐基础UNet模型

### Q: 知识蒸馏的教师模型如何选择？

A:
- **默认配置**: 系统自动使用所有8个基础模型作为教师
- **自定义选择**: 建议选择性能互补的3-5个模型
- **预训练策略**: 默认启用教师模型预训练，可使用`--no-pretrain-teachers`禁用

### Q: 如何处理内存不足问题？

A:
- 减小`--batch_size`（推荐值：1-2）
- 减小`--roi_size`（如[96,96,96]）
- 启用`--auto_adjust`自动调整参数
- 使用梯度累积技术

### Q: 评估指标如何解读？

A:
- **Dice > 0.8**: 优秀的分割性能
- **Dice 0.6-0.8**: 良好的分割性能
- **Dice < 0.6**: 需要改进的分割性能
- **Hausdorff距离**: 越小越好，表示边界越准确
- **FROC AUC**: 越大越好，表示检测性能越强

### Q: 如何提升模型性能？

A:
1. **数据增强**: 使用更多的数据增强策略
2. **模型集成**: 使用融合网络或知识蒸馏
3. **超参数调优**: 调整学习率、批次大小等
4. **预训练**: 使用预训练的教师模型
5. **架构搜索**: 使用NAS寻找最优架构

## 更新日志

### v2.0.0 (最新版本)
- ✅ 新增NAS-蒸馏集成功能
- ✅ 完善7种评估指标体系
- ✅ 优化自适应损失函数策略
- ✅ 增强智能重复检测机制
- ✅ 改进可视化图表生成
- ✅ 支持MS_MultiSpine数据集

### v1.5.0
- ✅ 新增融合网络架构
- ✅ 实现多教师知识蒸馏
- ✅ 添加神经架构搜索
- ✅ 完善评估指标体系

### v1.0.0
- ✅ 基础模型训练功能
- ✅ 支持8种基础模型架构
- ✅ 基础评估和推理功能

## 技术支持

如有问题或建议，请通过以下方式联系：

- 📧 邮箱: support@example.com
- 🐛 问题反馈: GitHub Issues
- 📖 文档: 项目Wiki页面

---

**注意**: 本项目基于MONAI框架开发，请确保正确安装所有依赖项。建议在虚拟环境中运行以避免依赖冲突。

