import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple, Any
import logging
import os
import json
from pathlib import Path

from nas_search import SuperNet, DARTSSearcher
from knowledge_distillation import KnowledgeDistillationLoss, MultiTeacherDistillation
from model import ModelFactory

class NASDistillationIntegration:
    """
    NAS-蒸馏集成类：实现两阶段训练流程
    阶段1：NAS搜索找到最优架构
    阶段2：使用最优架构作为学生模型进行知识蒸馏训练
    """
    
    def __init__(self,
                 teacher_models: List[str],
                 device: torch.device,
                 dataset_type: str = 'BraTS',
                 nas_epochs: int = 50,
                 distillation_epochs: int = 100,
                 arch_lr: float = 3e-4,
                 model_lr: float = 1e-3,
                 distillation_lr: float = 1e-4,
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 save_dir: str = './checkpoints/nas_distillation'):
        """
        初始化NAS-蒸馏集成器
        
        Args:
            teacher_models: 教师模型列表
            device: 计算设备
            dataset_type: 数据集类型
            nas_epochs: NAS搜索轮数
            distillation_epochs: 知识蒸馏轮数
            arch_lr: 架构参数学习率
            model_lr: 模型参数学习率
            distillation_lr: 蒸馏阶段学习率
            temperature: 蒸馏温度
            alpha: 蒸馏损失权重
            save_dir: 模型保存目录
        """
        self.teacher_models = teacher_models
        self.device = device
        self.dataset_type = dataset_type
        self.nas_epochs = nas_epochs
        self.distillation_epochs = distillation_epochs
        self.arch_lr = arch_lr
        self.model_lr = model_lr
        self.distillation_lr = distillation_lr
        self.temperature = temperature
        self.alpha = alpha
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 设置数据集相关参数
        if dataset_type == 'MS_MultiSpine':
            self.in_channels = 2
            self.num_classes = 6
        else:  # BraTS
            self.in_channels = 4
            self.num_classes = 4
            
        # 初始化组件
        self.supernet = None
        self.searcher = None
        self.teacher_ensemble = None
        self.student_model = None
        self.distillation_model = None
        
        # 训练状态
        self.nas_completed = False
        self.best_architecture = None
        
        logging.info(f"初始化NAS-蒸馏集成器: 教师模型={teacher_models}, 数据集={dataset_type}")
        
    def initialize_nas_search(self) -> None:
        """
        初始化NAS搜索组件
        """
        logging.info("初始化NAS搜索组件...")
        
        # 创建超网络
        self.supernet = SuperNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=32,
            num_layers=4,
            dataset_type=self.dataset_type
        )
        
        # 创建DARTS搜索器
        self.searcher = DARTSSearcher(
            supernet=self.supernet,
            device=self.device,
            arch_lr=self.arch_lr,
            model_lr=self.model_lr
        )
        
        logging.info("NAS搜索组件初始化完成")
        
    def initialize_teacher_models(self) -> None:
        """
        初始化教师模型集合
        """
        logging.info("初始化教师模型集合...")
        
        # 创建模型工厂
        model_factory = ModelFactory()
        
        # 加载教师模型
        teachers = {}
        for model_name in self.teacher_models:
            try:
                model = model_factory.create_model(
                    model_name=model_name,
                    in_channels=self.in_channels,
                    num_classes=self.num_classes,
                    dataset_type=self.dataset_type
                )
                model = model.to(self.device)
                model.eval()  # 教师模型设为评估模式
                teachers[model_name] = model
                logging.info(f"成功加载教师模型: {model_name}")
            except Exception as e:
                logging.warning(f"加载教师模型 {model_name} 失败: {e}")
                
        if not teachers:
            raise ValueError("没有成功加载任何教师模型")
            
        self.teacher_ensemble = teachers
        logging.info(f"教师模型集合初始化完成，共加载 {len(teachers)} 个模型")
        
    def search_architecture(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        执行NAS架构搜索
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            最优架构信息
        """
        if not self.supernet or not self.searcher:
            self.initialize_nas_search()
            
        logging.info(f"开始NAS架构搜索，共 {self.nas_epochs} 轮...")
        
        best_val_loss = float('inf')
        best_arch_params = None
        
        for epoch in range(self.nas_epochs):
            # 训练阶段
            self.supernet.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                # 更新模型参数
                self.searcher.model_optimizer.zero_grad()
                output = self.supernet(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.searcher.model_optimizer.step()
                
                # 更新架构参数（每隔几个batch）
                if batch_idx % 5 == 0:
                    self.searcher.arch_optimizer.zero_grad()
                    output = self.supernet(data)
                    arch_loss = F.cross_entropy(output, target)
                    arch_loss.backward()
                    self.searcher.arch_optimizer.step()
                
                train_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    logging.info(f"NAS Epoch {epoch+1}/{self.nas_epochs}, "
                               f"Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            # 验证阶段
            self.supernet.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.supernet(data)
                    val_loss += F.cross_entropy(output, target).item()
            
            val_loss /= len(val_loader)
            
            # 保存最优架构
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_arch_params = {name: param.clone() for name, param in self.supernet.arch_parameters.items()}
                
                # 保存最优架构
                arch_save_path = self.save_dir / 'best_architecture.pth'
                torch.save({
                    'arch_parameters': best_arch_params,
                    'val_loss': best_val_loss,
                    'epoch': epoch
                }, arch_save_path)
                
            logging.info(f"NAS Epoch {epoch+1}/{self.nas_epochs} 完成, "
                       f"训练损失: {train_loss/len(train_loader):.4f}, "
                       f"验证损失: {val_loss:.4f}, 最优验证损失: {best_val_loss:.4f}")
        
        # 保存搜索结果
        self.best_architecture = {
            'arch_parameters': best_arch_params,
            'val_loss': best_val_loss,
            'supernet_state': self.supernet.state_dict()
        }
        
        self.nas_completed = True
        logging.info(f"NAS架构搜索完成，最优验证损失: {best_val_loss:.4f}")
        
        return self.best_architecture
        
    def create_student_model(self) -> nn.Module:
        """
        基于搜索到的最优架构创建学生模型
        
        Returns:
            学生模型
        """
        if not self.nas_completed or not self.best_architecture:
            raise ValueError("必须先完成NAS搜索才能创建学生模型")
            
        logging.info("基于最优架构创建学生模型...")
        
        # 创建新的SuperNet实例作为学生模型
        student = SuperNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=32,
            num_layers=4,
            dataset_type=self.dataset_type
        )
        
        # 加载最优架构参数
        student.arch_parameters.load_state_dict(self.best_architecture['arch_parameters'])
        
        # 冻结架构参数，只训练模型参数
        for param in student.arch_parameters.parameters():
            param.requires_grad = False
            
        self.student_model = student.to(self.device)
        logging.info("学生模型创建完成")
        
        return self.student_model
        
    def initialize_distillation(self) -> None:
        """
        初始化知识蒸馏组件
        """
        if not self.teacher_ensemble:
            self.initialize_teacher_models()
            
        if not self.student_model:
            self.create_student_model()
            
        logging.info("初始化知识蒸馏组件...")
        
        # 创建多教师蒸馏模型
        self.distillation_model = MultiTeacherDistillation(
            teachers=self.teacher_ensemble,
            student=self.student_model,
            temperature=self.temperature,
            alpha=self.alpha,
            freeze_teachers=True
        )
        
        self.distillation_model = self.distillation_model.to(self.device)
        logging.info("知识蒸馏组件初始化完成")
        
    def distillation_training(self, train_loader, val_loader) -> Dict[str, float]:
        """
        执行知识蒸馏训练
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            训练结果统计
        """
        if not self.distillation_model:
            self.initialize_distillation()
            
        logging.info(f"开始知识蒸馏训练，共 {self.distillation_epochs} 轮...")
        
        # 优化器
        optimizer = torch.optim.Adam(
            self.distillation_model.student.parameters(),
            lr=self.distillation_lr,
            weight_decay=1e-4
        )
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.distillation_epochs
        )
        
        best_val_loss = float('inf')
        training_stats = {
            'train_losses': [],
            'val_losses': [],
            'best_epoch': 0
        }
        
        for epoch in range(self.distillation_epochs):
            # 训练阶段
            self.distillation_model.train()
            train_loss = 0.0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # 前向传播
                student_output = self.distillation_model.student(data)
                
                # 计算蒸馏损失
                distillation_loss = self.distillation_model.compute_distillation_loss(
                    data, target
                )
                
                distillation_loss.backward()
                optimizer.step()
                
                train_loss += distillation_loss.item()
                
                if batch_idx % 10 == 0:
                    logging.info(f"蒸馏 Epoch {epoch+1}/{self.distillation_epochs}, "
                               f"Batch {batch_idx}, Loss: {distillation_loss.item():.4f}")
            
            # 验证阶段
            self.distillation_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.to(self.device), target.to(self.device)
                    val_loss += self.distillation_model.compute_distillation_loss(
                        data, target
                    ).item()
            
            val_loss /= len(val_loader)
            train_loss /= len(train_loader)
            
            training_stats['train_losses'].append(train_loss)
            training_stats['val_losses'].append(val_loss)
            
            # 保存最优模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                training_stats['best_epoch'] = epoch
                
                # 保存最优学生模型
                student_save_path = self.save_dir / 'best_student_model.pth'
                torch.save({
                    'model_state_dict': self.student_model.state_dict(),
                    'arch_parameters': self.best_architecture['arch_parameters'],
                    'val_loss': val_loss,
                    'epoch': epoch,
                    'training_config': {
                        'teacher_models': self.teacher_models,
                        'dataset_type': self.dataset_type,
                        'temperature': self.temperature,
                        'alpha': self.alpha
                    }
                }, student_save_path)
                
            scheduler.step()
            
            logging.info(f"蒸馏 Epoch {epoch+1}/{self.distillation_epochs} 完成, "
                       f"训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}, "
                       f"最优验证损失: {best_val_loss:.4f}")
        
        training_stats['best_val_loss'] = best_val_loss
        
        # 保存训练统计
        stats_save_path = self.save_dir / 'training_stats.json'
        with open(stats_save_path, 'w') as f:
            json.dump(training_stats, f, indent=2)
            
        logging.info(f"知识蒸馏训练完成，最优验证损失: {best_val_loss:.4f}")
        
        return training_stats
        
    def full_training_pipeline(self, train_loader, val_loader) -> Dict[str, Any]:
        """
        执行完整的两阶段训练流程
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            
        Returns:
            完整训练结果
        """
        logging.info("开始NAS-蒸馏集成完整训练流程...")
        
        results = {}
        
        # 阶段1：NAS架构搜索
        logging.info("=== 阶段1：NAS架构搜索 ===")
        nas_results = self.search_architecture(train_loader, val_loader)
        results['nas_search'] = nas_results
        
        # 阶段2：知识蒸馏训练
        logging.info("=== 阶段2：知识蒸馏训练 ===")
        distillation_results = self.distillation_training(train_loader, val_loader)
        results['distillation'] = distillation_results
        
        # 保存完整结果
        results_save_path = self.save_dir / 'full_training_results.json'
        with open(results_save_path, 'w') as f:
            # 转换tensor为可序列化格式
            serializable_results = self._make_serializable(results)
            json.dump(serializable_results, f, indent=2)
            
        logging.info("NAS-蒸馏集成完整训练流程完成")
        
        return results
        
    def load_trained_model(self, model_path: str) -> nn.Module:
        """
        加载训练好的学生模型
        
        Args:
            model_path: 模型文件路径
            
        Returns:
            加载的学生模型
        """
        logging.info(f"加载训练好的学生模型: {model_path}")
        
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 创建学生模型
        student = SuperNet(
            in_channels=self.in_channels,
            num_classes=self.num_classes,
            base_channels=32,
            num_layers=4,
            dataset_type=self.dataset_type
        )
        
        # 加载架构参数
        student.arch_parameters.load_state_dict(checkpoint['arch_parameters'])
        
        # 加载模型权重
        student.load_state_dict(checkpoint['model_state_dict'])
        
        student = student.to(self.device)
        student.eval()
        
        logging.info("学生模型加载完成")
        
        return student
        
    def _make_serializable(self, obj: Any) -> Any:
        """
        将包含tensor的对象转换为可序列化格式
        """
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        else:
            return obj
            
    def get_save_paths(self) -> Dict[str, str]:
        """
        获取所有保存路径
        
        Returns:
            保存路径字典
        """
        return {
            'best_architecture': str(self.save_dir / 'best_architecture.pth'),
            'best_student_model': str(self.save_dir / 'best_student_model.pth'),
            'training_stats': str(self.save_dir / 'training_stats.json'),
            'full_results': str(self.save_dir / 'full_training_results.json'),
            'save_directory': str(self.save_dir)
        }


if __name__ == "__main__":
    # 测试代码
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    
    # 创建模拟数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 模拟BraTS数据
    batch_size = 2
    data = torch.randn(batch_size * 10, 4, 32, 32, 32)
    targets = torch.randint(0, 4, (batch_size * 10, 16, 16, 16))  # 匹配模型输出尺寸
    
    dataset = TensorDataset(data, targets)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 创建NAS-蒸馏集成器
    teacher_models = ['UNet3D', 'VNet3D', 'ResUNet3D']
    
    nas_distillation = NASDistillationIntegration(
        teacher_models=teacher_models,
        device=device,
        dataset_type='BraTS',
        nas_epochs=5,  # 测试用较少轮数
        distillation_epochs=5,
        save_dir='./test_checkpoints/nas_distillation'
    )
    
    try:
        # 执行完整训练流程
        results = nas_distillation.full_training_pipeline(train_loader, val_loader)
        print("NAS-蒸馏集成测试成功！")
        print(f"保存路径: {nas_distillation.get_save_paths()}")
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()