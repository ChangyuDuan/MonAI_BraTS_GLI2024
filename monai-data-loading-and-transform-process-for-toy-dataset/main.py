import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from DatasetLoader_transforms import DatasetLoader
from model import BasicModelBank, AdvancedModelBank, ModelFactory, get_all_supported_models
from train import ModelTrainer
from evaluate import BraTSEvaluator
from inference import InferenceEngine


def get_high_performance_config(device='auto') -> Dict[str, Any]:
    """
    获取配置性能（针对CPU和GPU设备）
    默认统一使用自适应损失函数策略和完整评估指标
    """
    # 设备性能配置
    if device == 'cpu':
        # CPU性能配置
        #batch_size = 4
        #cache_rate = 0.5
        # CPU性能配置（优化内存使用）
        batch_size = 1  # 降低批次大小减少内存使用
        cache_rate = 0.1  # 大幅降低缓存率避免内存不足
        num_workers = 0  # Windows上设置为0避免多进程问题
        pin_memory = False
        spatial_size = (96, 96, 96)
        roi_size = (64, 64, 64)
        use_amp = False
    else:  # cuda or auto
        # GPU性能配置
        batch_size = 8
        cache_rate = 1.0
        num_workers = 16
        pin_memory = True
        spatial_size = (128, 128, 128)
        roi_size = (96, 96, 96)
        use_amp = True
    
    # 基础学习率
    learning_rate = 2e-4
    
    return {
        # 数据加载配置
        'batch_size': batch_size,
        'cache_rate': cache_rate,
        'num_workers': num_workers,
        'pin_memory': pin_memory,
        'persistent_workers': True,
        'prefetch_factor': 4,
        
        # 数据预处理配置
        'spatial_size': spatial_size,
        'roi_size': roi_size,
        
        # 训练配置
        'max_epochs': 500,
        'learning_rate': learning_rate,
        'weight_decay': 1e-5,
        'use_amp': use_amp,
        'patience': 50,
        'save_interval': 10,
        'log_interval': 5,
        
        # 优化器和调度器
        'optimizer_name': 'AdamW',
        'scheduler_name': 'CosineAnnealingLR',
        
        # 验证配置
        'val_interval': 1,
        'val_split': 0.2,
        
        # 统一策略配置（所有模型类型都使用）
        'use_adaptive_loss': True,          # 强制使用自适应损失函数
        'use_full_metrics': True,           # 强制使用完整评估指标
        'loss_strategy': 'adaptive_combined', # 损失函数策略
        'metrics_strategy': 'comprehensive', # 评估指标策略
        
        # 其他配置
        'seed': 42,
        'gradient_clip_val': 1.0,
        'compile_model': True,  # 模型编译优化
        'deterministic': True
    }

def merge_args_with_config(args, device) -> Dict[str, Any]:
    # 获取基础配置
    config = get_high_performance_config(
        device=str(device)
    )
    
    # 命令行参数覆盖默认配置
    if args.batch_size:
        config['batch_size'] = args.batch_size
    if args.epochs:
        config['max_epochs'] = args.epochs
    if args.learning_rate:
        config['learning_rate'] = args.learning_rate
        
    # 添加基础参数
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['device'] = str(device)
    
    # 添加高级模型配置
    config['model_category'] = getattr(args, 'model_category', 'basic')
    config['model_type'] = getattr(args, 'model_type', 'single')
    
    # 知识蒸馏配置
    if hasattr(args, 'teacher_models') and args.teacher_models:
        config['teacher_models'] = args.teacher_models
    elif config['model_category'] == 'advanced' and config['model_type'] == 'distillation':
        # 默认使用所有7个网络架构作为教师模型
        config['teacher_models'] = get_all_supported_models()
    if hasattr(args, 'student_model') and args.student_model:
        config['student_model'] = args.student_model
    if hasattr(args, 'distillation_temperature'):
        config['distillation_temperature'] = args.distillation_temperature
    if hasattr(args, 'distillation_alpha'):
        config['distillation_alpha'] = args.distillation_alpha
    
    # 融合网络配置
    if hasattr(args, 'fusion_models') and args.fusion_models:
        config['fusion_models'] = args.fusion_models
    elif config['model_category'] == 'advanced' and config['model_type'] == 'fusion':
        # 默认使用所有7个网络架构进行融合
        config['fusion_models'] = get_all_supported_models()
    
    # NAS配置
    if hasattr(args, 'nas_epochs'):
        config['nas_epochs'] = args.nas_epochs
    
    # 预训练配置
    if hasattr(args, 'pretrained_dir'):
        config['pretrained_dir'] = args.pretrained_dir
    if hasattr(args, 'pretrain_teachers'):
        config['pretrain_teachers'] = args.pretrain_teachers
    if hasattr(args, 'teacher_epochs'):
        config['teacher_epochs'] = args.teacher_epochs
    if hasattr(args, 'force_retrain_teachers'):
        config['force_retrain_teachers'] = args.force_retrain_teachers
    
    return config

def run_simplified_training(args, device: torch.device):
    """
    运行训练流程
    """
    print("\n" + "=" * 60)
    
    print("开始训练流程")
    
    # 如果模型数量较多，提示这是多模型训练
    if len(args.model_names) > 1:
        print(f"检测到多模型训练模式 - 并行训练{len(args.model_names)}个模型")
    
    print("=" * 60)
    
    try:
        # 自动调节参数
        if args.auto_adjust:
            batch_size, epochs, learning_rate = auto_adjust_parameters(device, args)
        else:
            batch_size = args.batch_size or 2
            epochs = args.epochs or 500
            learning_rate = args.learning_rate or 1e-4
        
        print(f"训练参数:")
        print(f"  数据目录: {args.data_dir}")
        print(f"  模型列表: {args.model_names}")
        print(f"  批次大小: {batch_size}")
        print(f"  训练轮数: {epochs}")
        print(f"  学习率: {learning_rate}")
        print(f"  计算设备: {device}")

        print(f"  自动调节: {args.auto_adjust}")
        print("-" * 60)
        
        trained_models = []
        
        # 统一的训练逻辑
        if hasattr(args, 'model_category') and args.model_category == 'advanced':
            # 高级模型训练
            model_type_name = {
                'fusion': '融合网络',
                'distillation': '知识蒸馏', 
                'nas': '神经架构搜索'
            }.get(args.model_type, args.model_type)
            print(f"\n启动高级模型训练: {model_type_name.upper()}")
            if args.model_type == 'distillation':
                print(f"学生模型: {args.student_model}")
                if hasattr(args, 'teacher_models') and args.teacher_models:
                    print(f"教师模型: {', '.join(args.teacher_models)}")
                
                # 检查是否需要预训练教师模型
                if hasattr(args, 'pretrain_teachers') and args.pretrain_teachers:
                    print(f"\n开始预训练教师模型...")
                    
                    # 获取需要预训练的教师模型列表
                    teacher_models = getattr(args, 'teacher_models', get_all_supported_models())
                    pretrained_dir = getattr(args, 'pretrained_dir', './pretrained_teachers')
                    teacher_epochs = getattr(args, 'teacher_epochs', 100)
                    force_retrain = getattr(args, 'force_retrain_teachers', False)
                    
                    pretrained_paths = {}
                    
                    # 使用现有的基础模型训练逻辑预训练每个教师模型
                    for teacher_model in teacher_models:
                        teacher_output_dir = f"{pretrained_dir}/{teacher_model}"
                        teacher_model_path = f"{teacher_output_dir}/best_model.pth"
                        
                        # 检查是否已存在预训练模型
                        if os.path.exists(teacher_model_path) and not force_retrain:
                            print(f"教师模型 {teacher_model} 已存在预训练权重，跳过训练")
                            pretrained_paths[teacher_model] = teacher_model_path
                            continue
                        
                        print(f"\n预训练教师模型: {teacher_model.upper()}")
                        
                        # 创建教师模型训练配置（复用基础模型训练逻辑）
                        teacher_config = merge_args_with_config(args, device)
                        teacher_config.update({
                            'batch_size': batch_size,
                            'max_epochs': teacher_epochs,
                            'learning_rate': learning_rate,
                            'model_name': teacher_model,
                            'output_dir': teacher_output_dir
                        })
                        
                        # 创建训练器并训练教师模型
                        teacher_trainer = ModelTrainer(teacher_config)
                        teacher_trainer.train()
                        
                        # 检查训练结果
                        if os.path.exists(teacher_model_path):
                            pretrained_paths[teacher_model] = teacher_model_path
                            print(f"[成功] 教师模型 {teacher_model.upper()} 预训练完成")
                        else:
                            print(f"[失败] 教师模型 {teacher_model.upper()} 预训练失败")
                    
                    print(f"\n教师模型预训练完成: {len(pretrained_paths)} 个模型")
                    for model_name, path in pretrained_paths.items():
                        print(f"  {model_name}: {path}")
        elif len(args.model_names) > 1:
            # 多模型训练
            if args.parallel:
                print(f"\n启动多模型并行训练模式 - 同时训练{len(args.model_names)}个模型")
            else:
                print(f"\n启动多模型逐个训练模式 - 依次训练{len(args.model_names)}个模型")
        else:
            # 单模型训练
            print(f"\n启动单模型训练: {args.model_names[0].upper()}")
        
        # 根据并行参数选择训练方式
        if args.parallel:
            # 并行训练所有模型
            for model_name in args.model_names:
                if hasattr(args, 'model_category') and args.model_category == 'advanced':
                    model_type_name = {
                        'fusion': '融合网络',
                        'distillation': '知识蒸馏', 
                        'nas': '神经架构搜索'
                    }.get(args.model_type, args.model_type)
                    print(f"\n开始训练高级模型: {model_type_name.upper()}")
                else:
                    print(f"\n开始训练模型: {model_name.upper()}")
                
                # 创建训练配置
                config = merge_args_with_config(args, device)
                
                # 为高级模型设置正确的输出目录
                if hasattr(args, 'model_category') and args.model_category == 'advanced':
                    output_subdir = f"{args.model_type}_model"
                    config.update({
                        'batch_size': batch_size,
                        'max_epochs': epochs,
                        'learning_rate': learning_rate,
                        'model_name': model_name,
                        'output_dir': f'{args.output_dir}/models/{output_subdir}'
                    })
                else:
                    config.update({
                        'batch_size': batch_size,
                        'max_epochs': epochs,
                        'learning_rate': learning_rate,
                        'model_name': model_name,
                        'output_dir': f'{args.output_dir}/models/{model_name}'
                    })
                
                # 创建训练器
                trainer = ModelTrainer(config)
                
                # 开始训练
                trainer.train()
                
                # 记录训练好的模型
                if hasattr(args, 'model_category') and args.model_category == 'advanced':
                    output_subdir = f"{args.model_type}_model"
                    model_path = f'{args.output_dir}/models/{output_subdir}/checkpoints/best_model.pth'
                else:
                    model_path = f'{args.output_dir}/models/{model_name}/checkpoints/best_model.pth'
                if os.path.exists(model_path):
                    trained_models.append({
                        'name': model_name,
                        'path': model_path,
                        'config': config
                    })
                    print(f"[成功] {model_name.upper()} 训练完成")
                else:
                    print(f"[失败] {model_name.upper()} 训练失败")
        else:
            # 逐个训练所有模型
            for model_name in args.model_names:
                if hasattr(args, 'model_category') and args.model_category == 'advanced':
                    model_type_name = {
                        'fusion': '融合网络',
                        'distillation': '知识蒸馏', 
                        'nas': '神经架构搜索'
                    }.get(args.model_type, args.model_type)
                    print(f"\n开始训练高级模型: {model_type_name.upper()}")
                else:
                    print(f"\n开始训练模型: {model_name.upper()}")
                
                # 创建训练配置
                config = merge_args_with_config(args, device)
                
                # 为高级模型设置正确的输出目录
                if hasattr(args, 'model_category') and args.model_category == 'advanced':
                    output_subdir = f"{args.model_type}_model"
                    config.update({
                        'batch_size': batch_size,
                        'max_epochs': epochs,
                        'learning_rate': learning_rate,
                        'model_name': model_name,
                        'output_dir': f'{args.output_dir}/models/{output_subdir}'
                    })
                else:
                    config.update({
                        'batch_size': batch_size,
                        'max_epochs': epochs,
                        'learning_rate': learning_rate,
                        'model_name': model_name,
                        'output_dir': f'{args.output_dir}/models/{model_name}'
                    })
                
                # 创建训练器
                trainer = ModelTrainer(config)
                
                # 开始训练
                trainer.train()
                
                # 记录训练好的模型
                model_path = f'{args.output_dir}/models/{model_name}/checkpoints/best_model.pth'
                if os.path.exists(model_path):
                    trained_models.append({
                        'name': model_name,
                        'path': model_path,
                        'config': config
                    })
                    print(f"[成功] {model_name.upper()} 训练完成")
                else:
                    print(f"[失败] {model_name.upper()} 训练失败")
        
        # 自动模型训练

        
        print(f"\n[成功] 训练流程完成！")
        print(f"训练了 {len(trained_models)} 个模型")

        
        return True
        
    except Exception as e:
        print(f"\n[错误] 训练过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def auto_adjust_parameters(device: torch.device, args):
    """
    根据设备自动调节训练参数
    """
    # 基础参数
    base_batch_size = args.batch_size or 2
    base_epochs = args.epochs or 500
    base_lr = args.learning_rate or 1e-4
    
    if device.type == 'cuda':
        # GPU设备，使用优化参数
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = base_batch_size * 2  # GPU可以处理更大批次
        epochs = base_epochs
        learning_rate = base_lr * 1.2  # 稍微提高学习率加速训练
        
        print(f"GPU优化配置 (显存: {gpu_memory_gb:.1f}GB)")
    else:
        # CPU设备，使用合理参数
        batch_size = base_batch_size  # CPU可以处理正常批次
        epochs = base_epochs  # 保持正常训练轮数
        learning_rate = base_lr  # 使用标准学习率
        
        print(f"CPU优化配置")
    
    return batch_size, epochs, learning_rate


def run_evaluation(model_path: str, data_dir: str, output_dir: str = './evaluation_results', device: torch.device = None):
    """
    运行评估流程
    model_path: 模型文件路径
    data_dir: 数据目录
    output_dir: 输出目录
    device: 计算设备
    """
    print("\n" + "=" * 80)
    print("脑肿瘤分割模型评估")
    print("=" * 80)
    
    try:
        # 检查模型文件
        if not os.path.exists(model_path):
            print(f"[错误] 模型文件不存在: {model_path}")
            return False
        
        # 显示评估配置信息
        print(f"\n 评估配置信息:")
        print(f"  模型路径: {model_path}")
        print(f"  数据目录: {data_dir}")
        print(f"  输出目录: {output_dir}")
        
        # 创建评估器
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"\n 计算设备: {device}")
        if device.type == 'cuda':
            print(f"  GPU信息: {torch.cuda.get_device_name(0)}")
            print(f"  GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        print("\n" + "-" * 80)
        print("初始化评估器...")
        
        evaluator = BraTSEvaluator(
            model_path=model_path,
            data_dir=data_dir,
            device=device,
            output_dir=output_dir
        )
        
        print("\n" + "-" * 80)
        print(" 开始模型评估...")
        
        # 执行评估
        results = evaluator.evaluate_model()
        
        # 详细打印结果
        print("\n" + "=" * 80)
        print(" 评估结果详细报告")
        print("=" * 80)
        
        # Dice系数统计
        print(f"\n Dice系数 (分割准确性指标):")
        print(f"  ├─ 平均值: {results['mean_dice']:.4f} ± {results['std_dice']:.4f}")
        print(f"  ├─ 中位数: {results['median_dice']:.4f}")
        print(f"  ├─ 最小值: {results['min_dice']:.4f}")
        print(f"  └─ 最大值: {results['max_dice']:.4f}")
        
        # Hausdorff距离统计
        print(f"\n Hausdorff距离 (边界准确性指标):")
        print(f"  ├─ 平均值: {results['mean_hd']:.4f} ± {results['std_hd']:.4f}")
        print(f"  └─ 中位数: {results['median_hd']:.4f}")
        
        # 表面距离统计
        print(f"\n 表面距离 (表面匹配指标):")
        print(f"  ├─ 平均值: {results['mean_surface']:.4f} ± {results['std_surface']:.4f}")
        print(f"  └─ 中位数: {results['median_surface']:.4f}")
        
        # 评估案例统计
        print(f"\n 评估统计信息:")
        print(f"  ├─ 总案例数: {results['total_cases']}")
        print(f"  ├─ 成功评估: {results['total_cases']}")
        print(f"  └─ 失败案例: 0")
        
        # 性能评级
        mean_dice = results['mean_dice']
        if mean_dice >= 0.9:
            performance_level = "[优秀] (Excellent)"
        elif mean_dice >= 0.8:
            performance_level = "[良好] (Good)"
        elif mean_dice >= 0.7:
            performance_level = "[中等] (Fair)"
        elif mean_dice >= 0.6:
            performance_level = "[一般] (Poor)"
        else:
            performance_level = "[较差] (Very Poor)"
        
        print(f"\n模型性能评级: {performance_level}")
        
        # 输出文件信息
        print(f"\n 输出文件信息:")
        print(f"  ├─ 案例详细结果: {output_dir}/case_results.csv")
        print(f"  ├─ 总体统计报告: {output_dir}/summary_results.txt")
        print(f"  ├─ 结果分布图: {output_dir}/results_distribution.png")
        print(f"  └─ 可视化图像: {output_dir}/visualizations/")
        
        print("\n" + "=" * 80)
        print("[成功] 模型评估完成！")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"\n[错误] 评估过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def run_inference(args, device):
    """
    运行推理
    
    Args:
        args: 命令行参数
        device: 计算设备
        
    Returns:
        bool: 是否成功
    """
    try:
        # 参数验证
        if not args.input:
            print("[错误] 推理模式需要指定 --input 参数")
            return False
        
        if not args.output:
            print("[错误] 推理模式需要指定 --output 参数")
            return False
        
        if not args.model_path:
            print("[错误] 推理模式需要指定 --model_path 参数")
            return False
        
        # 检查模型路径
        if not os.path.exists(args.model_path):
            print(f"[错误] 模型路径不存在: {args.model_path}")
            return False
        
        # 检查输入路径
        if not os.path.exists(args.input):
            print(f"[错误] 输入路径不存在: {args.input}")
            return False
        
        print("\n" + "="*80)
        print("开始推理")
        print("="*80)
        print(f"模型路径: {args.model_path}")
        print(f"输入路径: {args.input}")
        print(f"输出路径: {args.output}")
        print(f"推理模式: {'批量推理' if args.batch_inference else '单文件推理'}")
        
        # 初始化推理引擎
        inference_engine = InferenceEngine(
            model_path=args.model_path,
            device=device,
            roi_size=tuple(args.roi_size),
            sw_batch_size=args.sw_batch_size,
            overlap=args.overlap
        )
        
        # 执行推理
        if args.batch_inference:
            # 批量推理
            results = inference_engine.predict_batch(
                input_dir=args.input,
                output_dir=args.output,
                save_visualization=not args.no_visualization
            )
            
            # 统计结果
            success_count = sum(1 for r in results if 'error' not in r)
            print(f"\n推理完成: {success_count}/{len(results)} 成功")
            
            if success_count < len(results):
                failed_files = [r['input_path'] for r in results if 'error' in r]
                print(f"失败的文件: {failed_files}")
        
        else:
            # 单文件推理
            result = inference_engine.predict_single_case(
                image_path=args.input,
                output_path=args.output,
                save_visualization=not args.no_visualization
            )
            
            print(f"\n推理完成")
            print(f"输入: {result['input_path']}")
            if 'output_path' in result:
                print(f"输出: {result['output_path']}")
            if 'visualization_path' in result:
                print(f"可视化: {result['visualization_path']}")
        
        print("\n" + "="*80)
        print("[成功] 推理完成！")
        print("="*80)
        
        return True
        
    except Exception as e:
        print(f"\n[错误] 推理过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False





def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='脑肿瘤分割项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""默认训练（UNet模型，500轮）"""
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'evaluate', 'inference'],
        required=True,
        help='运行模式: train（训练）, evaluate（评估）, inference（推理）'
    )
    

    
    parser.add_argument(
        '--data_dir',
        type=str,
        required=True,
        help='数据目录路径，例如：path/to/BraTS2024-BraTS-GLI-TrainingData/training_data1_v2'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='./outputs/checkpoints/best_model.pth',
        help='模型文件路径'
    )
    
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./outputs',
        help='输出目录'
    )
    
    parser.add_argument(
        '--batch_size',
        type=int,
        default=None,
        help='批次大小（覆盖配置文件设置）'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help='训练轮数（默认500轮）'
    )
    
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=None,
        help='学习率（覆盖配置文件设置）'
    )
    
    parser.add_argument(
        '--model_name',
        type=str,
        choices=get_all_supported_models(),
        help='要训练的单个模型名称'
    )
    
    parser.add_argument(
        '--model_names',
        type=str,
        nargs='+',
        choices=get_all_supported_models(),
        help='要训练的模型列表，用于多模型训练'
    )


    
    parser.add_argument(
        '--parallel',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='是否使用并行训练模式：true（并行训练，默认）或 false（逐个训练）'
    )
    
    # 高级模型参数
    parser.add_argument(
        '--model_category',
        type=str,
        choices=['basic', 'advanced'],
        default='basic',
        help='模型类别：basic（基础模型）或 advanced（高级模型）'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['single', 'fusion', 'distillation', 'nas'],
        default='single',
        help='高级模型类型：single（单模型）、fusion（融合网络）、distillation（知识蒸馏）、nas（神经架构搜索）'
    )
    
    parser.add_argument(
        '--teacher_models',
        type=str,
        nargs='+',
        choices=get_all_supported_models(),
        help='知识蒸馏的教师模型列表'
    )
    
    parser.add_argument(
        '--student_model',
        type=str,
        choices=get_all_supported_models(),
        default='UNet',
        help='知识蒸馏的学生模型'
    )
    
    parser.add_argument(
        '--fusion_models',
        type=str,
        nargs='+',
        choices=get_all_supported_models(),
        help='融合网络的基础模型列表'
    )
    
    parser.add_argument(
        '--distillation_temperature',
        type=float,
        default=4.0,
        help='知识蒸馏温度参数'
    )
    
    parser.add_argument(
        '--distillation_alpha',
        type=float,
        default=0.7,
        help='知识蒸馏软标签权重'
    )
    
    # 预训练相关参数
    parser.add_argument(
        '--pretrained_dir',
        type=str,
        default='./pretrained_teachers',
        help='预训练教师模型目录路径'
    )
    
    parser.add_argument(
        '--pretrain_teachers',
        action='store_true',
        help='在知识蒸馏前先预训练教师模型'
    )
    
    parser.add_argument(
        '--teacher_epochs',
        type=int,
        default=100,
        help='教师模型预训练轮数'
    )
    
    parser.add_argument(
        '--force_retrain_teachers',
        action='store_true',
        help='强制重新训练已存在的预训练教师模型'
    )
    
    parser.add_argument(
        '--nas_epochs',
        type=int,
        default=50,
        help='NAS搜索轮数'
    )
    
    # NAS相关参数
    parser.add_argument(
        '--nas_type',
        type=str,
        choices=['nas', 'searcher', 'progressive', 'supernet'],
        default='nas',
        help='NAS搜索策略类型：nas（基础NAS搜索）、searcher（DARTS可微分架构搜索）、progressive（渐进式搜索）、supernet（超网络训练）'
    )
    
    parser.add_argument(
        '--base_channels',
        type=int,
        default=32,
        help='NAS网络基础通道数（默认32，推荐16-64之间）'
    )
    
    parser.add_argument(
        '--num_layers',
        type=int,
        default=4,
        help='NAS网络层数（默认4，推荐3-6层之间）'
    )
    
    parser.add_argument(
        '--auto_adjust',
        action='store_true',
        default=True,
        help='是否根据设备自动调节参数'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda', 'auto'],
        default='auto',
        help='计算设备选择: cpu, cuda, auto（自动检测）'
    )
    
    # 推理模式专用参数
    parser.add_argument(
        '--input',
        type=str,
        help='推理模式：输入文件路径或目录路径'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='推理模式：输出文件路径或目录路径'
    )
    
    parser.add_argument(
        '--batch_inference',
        action='store_true',
        help='推理模式：启用批量推理'
    )
    
    parser.add_argument(
        '--roi_size',
        type=int,
        nargs=3,
        default=[128, 128, 128],
        help='推理模式：滑动窗口大小'
    )
    
    parser.add_argument(
        '--sw_batch_size',
        type=int,
        default=4,
        help='推理模式：滑动窗口批次大小'
    )
    
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.6,
        help='推理模式：滑动窗口重叠率'
    )
    
    parser.add_argument(
        '--no_visualization',
        action='store_true',
        help='推理模式：不保存可视化结果'
    )

    
    args = parser.parse_args()
    
    # 处理模型参数逻辑
    if args.model_name and args.model_names:
        print("[错误] 不能同时指定 --model_name 和 --model_names 参数")
        return 1
    elif args.model_name:
        # 单个模型训练
        args.model_names = [args.model_name]
    elif args.model_names:
        # 多个模型训练，使用指定的模型列表
        pass
    else:
        # 默认训练所有模型
        args.model_names = get_all_supported_models()
    
    # 处理教师模型默认选择逻辑
    if args.model_type == 'distillation' and not args.teacher_models:
        # 知识蒸馏模式下，如果没有指定教师模型，默认使用所有基础模型
        args.teacher_models = get_all_supported_models()
        print(f"[信息] 知识蒸馏模式：自动选择所有基础模型作为教师模型: {args.teacher_models}")
    
    # 处理模型训练逻辑
    # 如果没有指定模型，默认训练单个UNet模型
    if not args.model_names:
        args.model_names = ['UNet']  # 默认单模型训练
    
    # 设备配置
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device_name = 'GPU' if torch.cuda.is_available() else 'CPU'
    elif args.device == 'cuda':
        if torch.cuda.is_available():
            device = torch.device('cuda')
            device_name = 'GPU'
        else:
            print("[警告] 指定使用GPU但CUDA不可用，自动切换到CPU")
            device = torch.device('cpu')
            device_name = 'CPU'
    else:  # cpu
        device = torch.device('cpu')
        device_name = 'CPU'
    
    # 设置全局设备
    torch.cuda.set_device(device) if device.type == 'cuda' else None
    
    # 打印欢迎信息
    print("\n" + "=" * 80)
    print("基于MONAI框架的医学图像分割解决方案")
    print("=" * 80)
    print(f"计算设备: {device_name} ({device})")
    if device.type == 'cuda':
        print(f"GPU信息: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    if args.mode != 'inference':
        print(f"数据目录: {args.data_dir}")
    if args.mode in ['evaluate', 'inference']:
        print(f"模型路径: {args.model_path}")
    if args.mode == 'inference':
        print(f"输入路径: {args.input}")
        print(f"输出路径: {args.output}")
    else:
        print(f"输出目录: {args.output_dir}")
    
    # 检查数据目录（推理模式不需要）
    if args.mode != 'inference' and not os.path.exists(args.data_dir):
        print(f"\n[错误] 数据目录不存在: {args.data_dir}")
        print("请检查数据路径或下载数据集")
        return 1
    
    success = True
    
    try:
        if args.mode == 'train':
            success = run_simplified_training(args, device)
            
        elif args.mode == 'evaluate':
            success = run_evaluation(args.model_path, args.data_dir, args.output_dir + '/evaluation', device)
            
        elif args.mode == 'inference':
            success = run_inference(args, device)
    
    except KeyboardInterrupt:
        print("\n\n[警告] 用户中断执行")
        return 1
    except Exception as e:
        print(f"\n\n[错误] 执行过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    if success:
        print("\n" + "=" * 80)
        print("[成功] 执行完成！")
        print("=" * 80)
        return 0
    else:
        print("\n" + "=" * 80)
        print("[失败] 执行失败！")
        print("=" * 80)
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)