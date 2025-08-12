import os
import sys
import argparse
import torch
from pathlib import Path
from typing import Dict, Any
import warnings

# 设置CUDA内存分配策略以避免内存碎片
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

# 过滤PyTorch TorchScript相关的弃用警告
warnings.filterwarnings("ignore", message=".*TorchScript.*functional optimizers.*deprecated.*")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="torch.distributed.optim")

from DatasetLoader_transforms import DatasetLoader
from MSMultiSpineLoader import MSMultiSpineDatasetLoader, create_optimized_training_config
from model import BasicModelBank, get_all_supported_models
from train import ModelTrainer
from evaluate import ModelEvaluator
from inference import InferenceEngine

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 导入中文字体配置
try:
    from font_config import configure_chinese_font
    # 自动配置中文字体
    configure_chinese_font()
except ImportError:
    import warnings
    warnings.warn("未找到font_config模块，中文显示可能出现问题", UserWarning)

def get_high_performance_config(device='auto') -> Dict[str, Any]:
    """
    获取配置性能（针对CPU和GPU设备）
    默认统一使用自适应损失函数策略和完整评估指标
    """
    # 设备性能配置
    if device == 'cpu':
        batch_size = 4  
        cache_rate = 0.5  
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
         spatial_size = (160, 160, 160)
         roi_size = (128, 128, 128)
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
        'use_optimization': True,           # 默认启用优化策略（仅对MS_MultiSpine数据集生效）
        'loss_strategy': 'adaptive_combined', # 损失函数策略
        'metrics_strategy': 'comprehensive', # 评估指标策略
        
        # 其他配置
        'seed': 42,
        'gradient_clip_val': 1.0,
        'compile_model': True,  # 模型编译优化
        'deterministic': True,
        
        # 知识蒸馏默认配置
        'distillation_type': 'multi_teacher',     # 默认蒸馏类型
        'distillation_temperature': 4.0,          # 默认蒸馏温度
        'distillation_alpha': 0.7,                # 默认蒸馏损失权重
    }

def detect_dataset_type(data_dir: str) -> str:
    """
    自动检测数据集类型
    """
    try:
        # 检查是否为MS_MultiSpine数据集
        if os.path.exists(os.path.join(data_dir, 'MS_MultiSpine_dataset')):
            return 'MS_MultiSpine'
        
        # 检查目录中的文件结构
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        # MS_MultiSpine数据集特征：包含sub-xxx格式的目录
        ms_pattern_count = sum(1 for d in subdirs if d.startswith('sub-') and d[4:].isdigit())
        if ms_pattern_count > 0:
            # 进一步检查是否包含MS特有的文件
            for subdir in subdirs[:3]:  # 检查前3个目录
                subdir_path = os.path.join(data_dir, subdir)
                files = os.listdir(subdir_path)
                # 检查是否包含T2和其他模态文件
                has_t2 = any('T2' in f for f in files)
                has_lesion_mask = any('LESIONMASK' in f for f in files)
                if has_t2 and has_lesion_mask:
                    return 'MS_MultiSpine'
        
        # BraTS数据集特征：包含BraTS格式的目录或文件
        brats_pattern_count = sum(1 for d in subdirs if 'BraTS' in d or 'brats' in d.lower())
        if brats_pattern_count > 0:
            return 'BraTS'
        
        # 默认返回BraTS
        return 'BraTS'
        
    except Exception as e:
        print(f"数据集类型检测失败: {e}，使用默认BraTS类型")
        return 'BraTS'

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
        
    # 数据集类型检测和配置
    dataset_type = getattr(args, 'dataset_type', None)
    if dataset_type is None or dataset_type == 'auto':
        # 自动检测数据集类型
        detected_type = detect_dataset_type(args.data_dir)
        print(f"自动检测数据集类型: {detected_type}")
        dataset_type = detected_type
    else:
        print(f"使用指定数据集类型: {dataset_type}")
        
    # 添加基础参数
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    config['device'] = str(device)
    config['dataset_type'] = dataset_type
    
    # 优化策略配置
    use_optimization = getattr(args, 'use_optimization', True)
    # 当dataset_type为MS_MultiSpine时，默认启用优化策略
    if dataset_type == 'MS_MultiSpine':
        config['use_optimization'] = use_optimization
        if use_optimization:
            print(f"✅ 已启用MS_MultiSpine数据集优化策略")
    else:
        config['use_optimization'] = False
        if use_optimization and dataset_type != 'MS_MultiSpine':
            print(f"⚠ 警告：优化策略仅支持MS_MultiSpine数据集，当前数据集类型为{dataset_type}，已禁用优化策略")
    
    # 添加复合架构模型配置
    config['model_category'] = getattr(args, 'model_category', 'basic')
    config['model_type'] = getattr(args, 'model_type', 'fusion')
    
    # 知识蒸馏配置
    if hasattr(args, 'teacher_models') and args.teacher_models:
        config['teacher_models'] = args.teacher_models
    elif config['model_category'] == 'advanced' and config['model_type'] == 'distillation':
        # 默认使用所有网络架构作为教师模型
        config['teacher_models'] = get_all_supported_models()
    if hasattr(args, 'student_model') and args.student_model:
        config['student_model'] = args.student_model
    if hasattr(args, 'distillation_temperature'):
        config['distillation_temperature'] = args.distillation_temperature
    if hasattr(args, 'distillation_alpha'):
        config['distillation_alpha'] = args.distillation_alpha
    if hasattr(args, 'distillation_type'):
        config['distillation_type'] = args.distillation_type
    
    # 知识蒸馏参数验证：确保教师模型和学生模型不重复
    if config['model_category'] == 'advanced' and config['model_type'] == 'distillation':
        student_model = config.get('student_model', 'VNet3D')
        teacher_models = config.get('teacher_models', [])
        
        # 检查教师模型列表中是否包含学生模型
        if isinstance(teacher_models, list) and student_model in teacher_models:
            print(f"⚠ 警告：检测到教师模型列表中包含学生模型 '{student_model}'")
            # 自动从教师模型列表中移除学生模型
            config['teacher_models'] = [name for name in teacher_models if name != student_model]
            print(f"✓ 已自动从教师模型列表中移除学生模型，当前教师模型: {config['teacher_models']}")
            
            # 如果移除后教师模型列表为空，使用除学生模型外的所有模型
            if not config['teacher_models']:
                all_models = get_all_supported_models()
                config['teacher_models'] = [name for name in all_models if name != student_model]
                print(f"✓ 教师模型列表为空，自动使用除学生模型外的所有模型: {config['teacher_models']}")
        
        # 验证最终配置
        if not config.get('teacher_models'):
            raise ValueError("知识蒸馏需要至少一个教师模型")
        
        print(f"  知识蒸馏参数验证通过：")
        print(f"  学生模型: {student_model}")
        print(f"  教师模型: {config['teacher_models']}")
        print(f"  教师模型数量: {len(config['teacher_models'])}")
    
    # 融合网络配置
    if hasattr(args, 'fusion_models') and args.fusion_models:
        config['fusion_models'] = args.fusion_models
    elif config['model_category'] == 'advanced' and config['model_type'] == 'fusion':
        # 默认使用所有7个网络架构进行融合
        config['fusion_models'] = get_all_supported_models()
    
    # NAS配置
    if hasattr(args, 'nas_epochs'):
        config['nas_epochs'] = args.nas_epochs
    
    # NAS-蒸馏集成配置
    if hasattr(args, 'distillation_epochs'):
        config['distillation_epochs'] = args.distillation_epochs
    if hasattr(args, 'distillation_lr'):
        config['distillation_lr'] = args.distillation_lr
    if hasattr(args, 'nas_distillation_save_dir'):
        config['save_dir'] = args.nas_distillation_save_dir
    
    # 预训练配置
    if hasattr(args, 'pretrained_dir'):
        config['pretrained_dir'] = args.pretrained_dir
    # 处理预训练教师模型参数
    config['pretrain_teachers'] = getattr(args, 'pretrain_teachers', True)
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
        
        # 首先判断训练模式并显示相应信息
        if args.model_category == 'basic':
            # 基础模型训练模式判断
            if len(args.model_names) > 1:
                # 多模型训练（基础模型）
                if args.parallel:
                    print(f"\n启动多模型并行训练模式 - 同时训练{len(args.model_names)}个基础模型")
                else:
                    print(f"\n启动多模型逐个训练模式 - 依次训练{len(args.model_names)}个基础模型")
            else:
                # 单模型训练（基础模型）
                print(f"\n启动单模型训练: {args.model_names[0].upper()}")
        elif args.model_category == 'advanced':
            # 复合架构模型训练模式判断和训练逻辑
            model_type_name = {
                'fusion': '融合网络',
                'distillation': '知识蒸馏', 
                'nas': '神经架构搜索',
                'nas_distillation': 'NAS-蒸馏集成'
            }.get(args.model_type, args.model_type)
            print(f"\n启动复合架构模型训练: {model_type_name.upper()}")
            
            # 复合架构模型训练
            if args.model_type == 'distillation':
                print(f"学生模型: {args.student_model}")
                if hasattr(args, 'teacher_models') and args.teacher_models:
                    print(f"教师模型: {', '.join(args.teacher_models)}")
            elif args.model_type == 'nas_distillation':
                print(f"NAS-蒸馏集成模式：")
                print(f"  NAS搜索轮数: {getattr(args, 'nas_epochs', 50)}")
                print(f"  知识蒸馏轮数: {getattr(args, 'distillation_epochs', 100)}")
                if hasattr(args, 'teacher_models') and args.teacher_models:
                    print(f"  教师模型: {', '.join(args.teacher_models)}")
                print(f"  保存目录: {getattr(args, 'nas_distillation_save_dir', './checkpoints/nas_distillation')}")
                
                # 检查是否需要预训练教师模型（默认启用）
                pretrain_enabled = getattr(args, 'pretrain_teachers', True)
                if pretrain_enabled:
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
                        
                        # 创建教师模型训练配置
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
        
        # 根据模型类别选择训练方式
        if args.model_category == 'advanced':
            # 复合架构模型训练
            model_type_name = {
                'fusion': '融合网络',
                'distillation': '知识蒸馏', 
                'nas': '神经架构搜索'
            }.get(args.model_type, args.model_type)
            print(f"\n开始训练复合架构模型: {model_type_name.upper()}")
            
            # 创建训练配置
            config = merge_args_with_config(args, device)
            output_subdir = f"{args.model_type}_model"
            config.update({
                'batch_size': batch_size,
                'max_epochs': epochs,
                'learning_rate': learning_rate,
                'output_dir': f'{args.output_dir}/models/{output_subdir}'
            })
            
            # 创建训练器
            trainer = ModelTrainer(config)
            trainer.train()
            
            # 检查训练结果
            model_path = f'{args.output_dir}/models/{output_subdir}/checkpoints/best_model.pth'
            if os.path.exists(model_path):
                trained_models.append({
                    'name': args.model_type,
                    'path': model_path,
                    'config': config
                })
                print(f"[成功] {model_type_name.upper()} 训练完成")
            else:
                print(f"[失败] {model_type_name.upper()} 训练失败")
        
        elif args.model_category == 'basic':
            # 基础模型训练
            if args.parallel:
                # 并行训练所有基础模型
                import threading
                import queue
                
                print(f"\n启动基础模型并行训练 - 同时训练{len(args.model_names)}个基础模型")
                
                # 创建结果队列和线程列表
                result_queue = queue.Queue()
                threads = []
                
                def train_model_thread(model_name, config, result_queue):
                    """单个模型训练的线程函数"""
                    try:
                        print(f"[线程] 开始训练模型: {model_name.upper()}")
                        trainer = ModelTrainer(config)
                        trainer.train()
                        
                        # 检查训练结果
                        model_path = f'{config["output_dir"]}/checkpoints/best_model.pth'
                        if os.path.exists(model_path):
                            result_queue.put({
                                'success': True,
                                'name': model_name,
                                'path': model_path,
                                'config': config
                            })
                            print(f"[线程] {model_name.upper()} 训练完成")
                        else:
                            result_queue.put({
                                'success': False,
                                'name': model_name,
                                'error': '模型文件未生成'
                            })
                            print(f"[线程] {model_name.upper()} 训练失败")
                    except Exception as e:
                        result_queue.put({
                            'success': False,
                            'name': model_name,
                            'error': str(e)
                        })
                        print(f"[线程] {model_name.upper()} 训练出错: {str(e)}")
                
                # 为每个模型创建训练线程
                for model_name in args.model_names:
                    config = merge_args_with_config(args, device)
                    config.update({
                        'batch_size': batch_size,
                        'max_epochs': epochs,
                        'learning_rate': learning_rate,
                        'model_name': model_name,
                        'output_dir': f'{args.output_dir}/models/{model_name}'
                    })
                    
                    thread = threading.Thread(
                        target=train_model_thread,
                        args=(model_name, config, result_queue),
                        name=f"Train-{model_name}"
                    )
                    threads.append(thread)
                    thread.start()
                    print(f"[主线程] 启动 {model_name.upper()} 训练线程")
                
                # 等待所有线程完成
                print(f"\n等待所有{len(threads)}个训练线程完成...")
                for thread in threads:
                    thread.join()
                
                # 收集所有训练结果
                print(f"\n收集训练结果...")
                while not result_queue.empty():
                    result = result_queue.get()
                    if result['success']:
                        trained_models.append({
                            'name': result['name'],
                            'path': result['path'],
                            'config': result['config']
                        })
                        print(f"[成功] {result['name'].upper()} 并行训练完成")
                    else:
                        print(f"[失败] {result['name'].upper()} 并行训练失败: {result.get('error', '未知错误')}")
        else:
            # 逐个训练所有基础模型
            print(f"\n启动基础模型逐个训练 - 依次训练{len(args.model_names)}个基础模型")
            
            # 逐个训练每个模型
            for i, model_name in enumerate(args.model_names, 1):
                print(f"\n[{i}/{len(args.model_names)}] 开始训练基础模型: {model_name.upper()}")
                
                # 创建训练配置
                config = merge_args_with_config(args, device)
                config.update({
                    'model_name': model_name,
                    'batch_size': batch_size,
                    'max_epochs': epochs,
                    'learning_rate': learning_rate,
                    'output_dir': f'{args.output_dir}/models/{model_name}'
                })
                
                # 创建训练器
                trainer = ModelTrainer(config)
                trainer.train()
                
                # 检查训练结果
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
            
            print(f"\n基础模型逐个训练完成，共训练了 {len(trained_models)} 个模型")
        
        # 训练流程完成 
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


def run_evaluation(model_path: str = None, data_dir: str = None, output_dir: str = './evaluation_results', device: torch.device = None):
    """
    运行评估流程
    model_path: 模型文件路径
    data_dir: 数据目录
    output_dir: 输出目录
    device: 计算设备
    """
    print("\n" + "=" * 80)
    print("医学图像分割模型评估")
    print("=" * 80)
    
    try:
        # 如果未指定训练好的模型路径，将根据时间优先原则自动查找最新模型
        if model_path is None:
            outputs_dir = './outputs'
            if os.path.exists(outputs_dir):
                # 查找outputs目录下的所有模型目录
                model_dirs = [d for d in os.listdir(outputs_dir) 
                             if os.path.isdir(os.path.join(outputs_dir, d))]
                if model_dirs:
                    # 选择最新的模型目录
                    latest_model_dir = max(model_dirs, 
                                         key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)))
                    model_path = os.path.join(outputs_dir, latest_model_dir)
                    print(f"自动选择最新训练的模型: {model_path}")
                else:
                    print(f"[错误] 在 {outputs_dir} 目录中未找到任何训练好的模型")
                    return False
            else:
                print(f"[错误] 输出目录不存在: {outputs_dir}")
                return False
        
        # 检查模型路径
        if not os.path.exists(model_path):
            print(f"[错误] 模型路径不存在: {model_path}")
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
        
        evaluator = ModelEvaluator(
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
    """
    try:
        # 参数验证
        if not args.input:
            print("[错误] 推理模式需要指定 --input 参数")
            return False
        
        if not args.output:
            print("[错误] 推理模式需要指定 --output 参数")
            return False
        
        # 如果未指定训练好的模型路径，将根据时间优先原则自动查找最新模型
        if not hasattr(args, 'model_path') or not args.model_path:
            outputs_dir = './outputs'
            if os.path.exists(outputs_dir):
                # 查找outputs目录下的所有模型目录
                model_dirs = [d for d in os.listdir(outputs_dir) 
                             if os.path.isdir(os.path.join(outputs_dir, d))]
                if model_dirs:
                    # 选择最新的模型目录
                    latest_model_dir = max(model_dirs, 
                                         key=lambda x: os.path.getmtime(os.path.join(outputs_dir, x)))
                    args.model_path = os.path.join(outputs_dir, latest_model_dir)
                    print(f"自动选择最新训练的模型: {args.model_path}")
                else:
                    print(f"[错误] 在 {outputs_dir} 目录中未找到任何训练好的模型")
                    return False
            else:
                print(f"[错误] 输出目录不存在: {outputs_dir}")
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
        description='医学图像分割项目',
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
        help='数据目录路径，例如：path/to/medical_data/training_data'
    )
    
    parser.add_argument(
        '--dataset_type',
        type=str,
        choices=['BraTS', 'MS_MultiSpine', 'auto'],
        default='auto',
        help='数据集类型：BraTS（脑肿瘤分割数据集）、MS_MultiSpine（多发性硬化脊柱数据集）或 auto（自动检测）'
    )
    
    parser.add_argument(
        '--model_path',
        type=str,
        default='./outputs/models/',
        help='模型文件路径或模型目录路径（将自动查找对应模型类型下的best_model.pth）'
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
    
    # 模型类别参数（必需）
    parser.add_argument(
        '--model_category',
        type=str,
        choices=['basic', 'advanced'],
        required=True,
        help='模型类别：basic（基础模型,需指定--model_name或--model_names）或 advanced（复合架构模型,需指定--model_type）'
    )
    
    parser.add_argument(
        '--model_type',
        type=str,
        choices=['fusion', 'distillation', 'nas', 'nas_distillation'],
        default='fusion',
        help='复合架构模型类型：fusion（融合网络）、distillation（知识蒸馏）、nas（神经架构搜索）、nas_distillation（NAS-蒸馏集成）'
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
        default='VNet3D',
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
    
    parser.add_argument(
        '--distillation_type',
        type=str,
        choices=['multi_teacher', 'progressive'],
        default='multi_teacher',
        help='知识蒸馏类型：multi_teacher（多教师蒸馏）或 progressive（渐进式蒸馏）'
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
        type=bool,
        default=True,
        help='启用教师模型预训练（默认启用）'
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
    
    # NAS-蒸馏集成相关参数
    parser.add_argument(
        '--distillation_epochs',
        type=int,
        default=100,
        help='NAS-蒸馏集成模式：知识蒸馏训练轮数'
    )
    
    parser.add_argument(
        '--distillation_lr',
        type=float,
        default=1e-4,
        help='NAS-蒸馏集成模式：知识蒸馏阶段学习率'
    )
    
    parser.add_argument(
        '--nas_distillation_save_dir',
        type=str,
        default='./checkpoints/nas_distillation',
        help='NAS-蒸馏集成模式：模型保存目录'
    )
    
    # NAS相关参数
    parser.add_argument(
        '--nas_type',
        type=str,
        choices=['searcher', 'progressive', 'supernet'],
        default='supernet',
        help='NAS搜索策略类型：supernet（超网络训练）、searcher（DARTS可微分架构搜索）、progressive（渐进式搜索）'
    )
    
    #超网络参数（nas_type=supernet时/不指定使用默认值时使用）
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
    
    # DARTS搜索参数（nas_type=searcher时使用）
    parser.add_argument(
        '--arch_lr',
        type=float,
        default=3e-4,
        help='架构参数学习率（默认3e-4，推荐1e-4到5e-4之间）'
    )
    
    parser.add_argument(
        '--model_lr',
        type=float,
        default=1e-3,
        help='模型权重学习率（默认1e-3，推荐5e-4到2e-3之间）'
    )
    
    # 渐进式NAS参数（nas_type=progressive时使用）
    parser.add_argument(
        '--max_layers',
        type=int,
        default=8,
        help='最大网络层数（默认8，推荐4-10层之间）'
    )
    
    parser.add_argument(
        '--start_layers',
        type=int,
        default=2,
        help='起始网络层数（默认2，推荐2-4层开始）'
    )

    
    parser.add_argument(
        '--use_optimization',
        type=lambda x: x.lower() == 'true',
        default=True,
        help='是否启用优化策略（默认True，当dataset_type为MS_MultiSpine时自动启用优化策略）'
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
    if args.model_category == 'basic':
        # 基础模型训练参数验证
        if args.model_name and args.model_names:
            print("[错误] 不能同时指定 --model_name 和 --model_names 参数")
            return 1
        elif args.model_name:
            # 单个基础模型训练
            args.model_names = [args.model_name]
        elif args.model_names:
            # 多个基础模型训练，使用指定的模型列表
            pass
        else:
            print("[错误] 基础模型训练必须指定 --model_name 或 --model_names 参数")
            return 1
    elif args.model_category == 'advanced':
        # 复合架构模型训练参数验证
        if not args.model_type:
            print("[错误] 复合架构模型训练必须指定有效的 --model_type 参数 (fusion/distillation/nas)")
            return 1
        # 复合架构模型不使用 model_names
        args.model_names = []
    
    # 处理教师模型默认选择逻辑
    if args.model_type == 'distillation' and not args.teacher_models:
        # 知识蒸馏模式下，如果没有指定教师模型，默认使用所有基础模型
        args.teacher_models = get_all_supported_models()
        print(f"[信息] 知识蒸馏模式：自动选择所有基础模型作为教师模型: {args.teacher_models}")
    
    # 处理NAS-蒸馏集成模式的教师模型默认选择逻辑
    if args.model_type == 'nas_distillation' and not args.teacher_models:
        # NAS-蒸馏集成模式下，如果没有指定教师模型，默认使用所有基础模型
        args.teacher_models = get_all_supported_models()
        print(f"[信息] NAS-蒸馏集成模式：自动选择所有基础模型作为教师模型: {args.teacher_models}")
    
    # 处理模型训练逻辑
    # 如果没有指定模型，默认训练单个VNet3D模型
    if not args.model_names:
        args.model_names = ['VNet3D']  # 默认单模型训练
    
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
    if device.type == 'cuda':
        torch.cuda.set_device(device.index if device.index is not None else 0)
    
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