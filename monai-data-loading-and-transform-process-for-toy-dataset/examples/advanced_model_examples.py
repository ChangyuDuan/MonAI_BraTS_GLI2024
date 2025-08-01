#!/usr/bin/env python3
"""
高级模型使用示例

本文件展示了如何使用新实现的高级模型功能，包括：
1. 知识蒸馏 (Knowledge Distillation)
2. 融合网络 (Fusion Networks)
3. 神经架构搜索 (Neural Architecture Search)

作者: AI Assistant
日期: 2024
"""

import os
import sys
import torch
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from model import ModelFactory, AdvancedModelBank
from train import ModelTrainer
from inference import InferenceEngine

def example_knowledge_distillation():
    """
    知识蒸馏示例
    使用多个教师模型训练一个轻量级学生模型
    """
    print("\n" + "=" * 60)
    print("知识蒸馏示例")
    print("=" * 60)
    
    # 配置参数
    config = {
        'model_category': 'advanced',
        'model_type': 'distillation',
        'teacher_models': ['UNet', 'SegResNet', 'UNETR'],  # 多个教师模型
        'student_model': 'UNet',  # 学生模型
        'distillation_temperature': 4.0,  # 蒸馏温度
        'distillation_alpha': 0.7,  # 软标签权重
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 2,
        'max_epochs': 100,
        'learning_rate': 1e-4,
        'data_dir': './data/training_data',
        'output_dir': './outputs/distillation_example'
    }
    
    print(f"教师模型: {config['teacher_models']}")
    print(f"学生模型: {config['student_model']}")
    print(f"蒸馏温度: {config['distillation_temperature']}")
    print(f"软标签权重: {config['distillation_alpha']}")
    
    try:
        # 创建知识蒸馏模型
        model = ModelFactory.create_model(
            model_type='distillation',
            config=config,
            device=config['device']
        )
        
        print(f"\n成功创建知识蒸馏模型")
        print(f"模型类型: {type(model).__name__}")
        
        # 创建训练器
        trainer = ModelTrainer(config)
        print("\n训练器创建成功，可以开始训练...")
        
        # 注意：实际训练需要真实数据
        # trainer.train()
        
    except Exception as e:
        print(f"知识蒸馏示例失败: {e}")

def example_fusion_network():
    """
    融合网络示例
    将多个不同架构的模型在特征级别进行融合
    """
    print("\n" + "=" * 60)
    print("融合网络示例")
    print("=" * 60)
    
    # 配置参数
    config = {
        'model_category': 'advanced',
        'model_type': 'fusion',
        'fusion_models': ['UNet', 'SegResNet', 'AttentionUNet'],  # 要融合的模型
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 2,
        'max_epochs': 200,
        'learning_rate': 2e-4,
        'data_dir': './data/training_data',
        'output_dir': './outputs/fusion_example'
    }
    
    print(f"融合模型: {config['fusion_models']}")
    print(f"设备: {config['device']}")
    
    try:
        # 创建融合网络
        model = ModelFactory.create_model(
            model_type='fusion',
            config=config,
            device=config['device']
        )
        
        print(f"\n成功创建融合网络")
        print(f"模型类型: {type(model).__name__}")
        
        # 创建训练器
        trainer = ModelTrainer(config)
        print("\n训练器创建成功，可以开始训练...")
        
        # 注意：实际训练需要真实数据
        # trainer.train()
        
    except Exception as e:
        print(f"融合网络示例失败: {e}")

def example_neural_architecture_search():
    """
    神经架构搜索示例
    自动搜索最优的网络架构
    """
    print("\n" + "=" * 60)
    print("神经架构搜索示例")
    print("=" * 60)
    
    # 配置参数
    config = {
        'model_category': 'advanced',
        'model_type': 'nas',
        'nas_epochs': 50,  # NAS搜索轮数
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'batch_size': 1,  # NAS通常需要较小的批次大小
        'max_epochs': 300,
        'learning_rate': 3e-4,
        'data_dir': './data/training_data',
        'output_dir': './outputs/nas_example'
    }
    
    print(f"NAS搜索轮数: {config['nas_epochs']}")
    print(f"设备: {config['device']}")
    print(f"批次大小: {config['batch_size']}")
    
    try:
        # 创建NAS模型
        model = ModelFactory.create_model(
            model_type='nas',
            config=config,
            device=config['device']
        )
        
        print(f"\n成功创建NAS模型")
        print(f"模型类型: {type(model).__name__}")
        
        # 创建训练器
        trainer = ModelTrainer(config)
        print("\n训练器创建成功，可以开始架构搜索...")
        
        # 注意：实际搜索需要真实数据
        # trainer.train()
        
    except Exception as e:
        print(f"神经架构搜索示例失败: {e}")

def example_advanced_inference():
    """
    高级模型推理示例
    展示如何使用训练好的高级模型进行推理
    """
    print("\n" + "=" * 60)
    print("高级模型推理示例")
    print("=" * 60)
    
    # 假设的模型路径
    model_paths = {
        'distillation': './outputs/distillation_example/best_model.pth',
        'fusion': './outputs/fusion_example/best_model.pth',
        'nas': './outputs/nas_example/best_model.pth'
    }
    
    for model_type, model_path in model_paths.items():
        print(f"\n--- {model_type.upper()} 模型推理 ---")
        
        if not os.path.exists(model_path):
            print(f"模型文件不存在: {model_path}")
            print("请先训练相应的模型")
            continue
            
        try:
            # 创建推理引擎
            inference_engine = InferenceEngine(
                model_path=model_path,
                device='auto',
                roi_size=(128, 128, 128),
                sw_batch_size=2
            )
            
            print(f"成功加载 {model_type} 模型")
            print(f"设备: {inference_engine.device}")
            
            # 示例推理（需要真实的输入文件）
            # result = inference_engine.predict_single_case(
            #     input_path='./data/test_case.nii.gz',
            #     output_path=f'./outputs/{model_type}_prediction.nii.gz'
            # )
            
        except Exception as e:
            print(f"{model_type} 模型推理失败: {e}")

def main():
    """
    主函数：运行所有示例
    """
    print("高级模型功能示例")
    print("本示例展示了知识蒸馏、融合网络和神经架构搜索的使用方法")
    print("注意：运行这些示例需要真实的训练数据")
    
    # 检查设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\n当前设备: {device}")
    
    if device == 'cpu':
        print("警告: 使用CPU运行，高级模型训练可能会很慢")
    
    # 运行示例
    try:
        example_knowledge_distillation()
        example_fusion_network()
        example_neural_architecture_search()
        example_advanced_inference()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成")
        print("=" * 60)
        
    except KeyboardInterrupt:
        print("\n用户中断执行")
    except Exception as e:
        print(f"\n示例运行失败: {e}")

if __name__ == "__main__":
    main()