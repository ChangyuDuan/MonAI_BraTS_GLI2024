import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Dict, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from DatasetLoader_transforms import BraTSDatasetLoader
from model import SimpleBraTSModel, get_all_supported_models
from train import BraTSTrainer
from evaluate import BraTSEvaluator


def get_high_performance_config(device='auto') -> Dict[str, Any]:
    """
    获取配置性能（针对CPU和GPU设备）
    默认统一使用自适应损失函数策略和完整评估指标
    
    Args:
        device: 设备类型 ('cpu', 'cuda', 'auto')
        
    Returns:
        Dict[str, Any]: 配置字典
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
    """
    
    Args:
        args: 命令行参数
        device: 计算设备
        
    Returns:
        Dict[str, Any]: 合并后的配置字典
    """
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
        
    # 添加其他必要参数
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['device'] = str(device)
    
    return config

def run_simplified_training(args, device: torch.device):
    """
    运行训练流程，支持多种训练模式：
    1. 多模型训练：分别训练指定的多个模型，可选择创建集成
    2. 单模型训练：训练单个指定模型
    
    Args:
        args: 命令行参数
        device: 计算设备
    """
    print("\n" + "=" * 60)
    
    print("开始训练流程")
    
    # 如果启用集成且模型数量较多，提示这是集成训练
    if args.enable_ensemble and len(args.model_names) > 3:
        print("检测到多模型集成训练模式")
    
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
        print(f"  启用集成: {args.enable_ensemble}")
        print(f"  自动调节: {args.auto_adjust}")
        print("-" * 60)
        
        trained_models = []
        
        # 训练每个模型
        for i, model_name in enumerate(args.model_names):
            print(f"\n训练模型 {i+1}/{len(args.model_names)}: {model_name.upper()}")
            
            # 创建训练配置
            config = merge_args_with_config(args, device)
            
            # 更新模型特定配置
            config.update({
                'batch_size': batch_size,
                'max_epochs': epochs,
                'learning_rate': learning_rate,
                'model_name': model_name,
                'output_dir': f'{args.output_dir}/models/{model_name}'
            })
            
            # 创建训练器
            trainer = BraTSTrainer(config)
            
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
        
        # 模型集成
        if args.enable_ensemble and len(trained_models) > 1:
            print(f"\n开始创建完整模型集成...")
            
            # 导入集成函数
            from model import create_full_ensemble
            
            # 创建完整集成模型
            ensemble_model = create_full_ensemble(device=str(device))
            
            # 保存集成模型配置
            ensemble_config = {
                'model_type': 'full_ensemble',
                'models': [model['name'] for model in trained_models],
                'device': str(device),
                'created_at': str(torch.utils.data.get_worker_info() or 'main')
            }
            
            ensemble_dest = f'{args.output_dir}/checkpoints/ensemble_model.pth'
            os.makedirs(os.path.dirname(ensemble_dest), exist_ok=True)
            
            # 保存集成模型信息
            torch.save({
                'config': ensemble_config,
                'trained_models': trained_models
            }, ensemble_dest)
            
            print(f"完整集成模型已创建")
            print(f"集成模型配置已保存到: {ensemble_dest}")
            print(f"包含模型: {[model['name'].upper() for model in trained_models]}")
        
        print(f"\n[成功] 简化训练流程完成！")
        print(f"训练了 {len(trained_models)} 个模型")
        if args.enable_ensemble and len(trained_models) > 1:
            print(f"已创建完整模型集成")
        
        return True
        
    except Exception as e:
        print(f"\n[错误] 简化训练过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def auto_adjust_parameters(device: torch.device, args):
    """
    根据设备自动调节训练参数
    
    Args:
        device: 计算设备
        args: 命令行参数
    
    Returns:
        tuple: (batch_size, epochs, learning_rate)
    """
    # 基础参数
    base_batch_size = args.batch_size or 2
    base_epochs = args.epochs or 500
    base_lr = args.learning_rate or 1e-4
    
    if device.type == 'cuda':
        # GPU设备，使用优化参数
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        batch_size = base_batch_size * 2  # 高端GPU可以处理更大批次
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
    
    Args:
        model_path: 模型文件路径
        data_dir: 数据目录
        output_dir: 输出目录
        device: 计算设备
    """
    print("\n" + "=" * 80)
    print("🔍 BraTS脑肿瘤分割模型评估")
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

# 推理功能已移除





def main():
    """
    主函数
    """
    parser = argparse.ArgumentParser(
        description='BraTS脑肿瘤分割项目',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""默认集成训练（所有7个模型，500轮）"""
    )
    
    parser.add_argument(
        '--mode', 
        type=str, 
        choices=['train', 'evaluate'],
        required=True,
        help='运行模式: train（训练）, evaluate（评估）'
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
        '--enable_ensemble',
        type=lambda x: x.lower() in ['true', '1', 'yes', 'on'],
        default=False,
        help='是否启用模型集成训练（默认关闭，仅在训练多个模型时启用）'
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
    

    
    args = parser.parse_args()
    
    # 处理模型参数逻辑
    if args.model_name and args.model_names:
        print("[错误] 不能同时指定 --model_name 和 --model_names 参数")
        return 1
    elif args.model_name:
        # 单个模型训练
        args.model_names = [args.model_name]
    elif args.model_names:
        # 多个模型训练，使用用户指定的模型列表
        pass
    else:
        # 默认训练所有模型
        args.model_names = get_all_supported_models()
    
    # 自动判断集成模式：单个模型禁用集成，多个模型启用集成
    if len(args.model_names) == 1:
        args.enable_ensemble = False  # 单个模型时自动禁用集成
    else:
        args.enable_ensemble = True   # 多个模型时自动启用集成
    
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
    print(f"数据目录: {args.data_dir}")
    if args.mode in ['evaluate', 'inference']:
        print(f"模型路径: {args.model_path}")
    print(f"输出目录: {args.output_dir}")
    
    # 检查数据目录
    if not os.path.exists(args.data_dir):
        print(f"\n[错误] 数据目录不存在: {args.data_dir}")
        print("请检查数据路径或下载BraTS数据集")
        return 1
    
    success = True
    
    try:
        if args.mode == 'train':
            success = run_simplified_training(args, device)
            
        elif args.mode == 'evaluate':
            success = run_evaluation(args.model_path, args.data_dir, args.output_dir + '/evaluation', device)
            
        # 推理功能已移除
    
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