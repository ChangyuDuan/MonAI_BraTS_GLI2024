import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional


class KnowledgeDistillationLoss(nn.Module):
    """
    知识蒸馏损失函数
    结合软标签(教师模型)和硬标签(真实标签)的损失
    """
    
    def __init__(self, 
                 temperature: float = 4.0,
                 alpha: float = 0.7,
                 beta: float = 0.3):
        """
        Args:
            temperature: 蒸馏温度参数
            alpha: 蒸馏损失权重
            beta: 真实标签损失权重
        """
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        计算知识蒸馏损失
        Args:
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            labels: 真实标签 
        Returns:
            总损失
        """
        # 处理标签维度：如果标签有额外的通道维度，去除它
        if len(labels.shape) == 5 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # 从 [B, 1, H, W, D] 变为 [B, H, W, D]
        
        # 确保标签是Long类型（整数），CrossEntropyLoss要求标签为整数类型
        labels = labels.long()
        
        # 软标签损失 (蒸馏损失)
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distillation_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # 硬标签损失 (真实标签损失)
        student_hard = student_logits
        hard_loss = self.ce_loss(student_hard, labels)
        
        # 组合损失
        total_loss = self.alpha * distillation_loss + self.beta * hard_loss
        
        return total_loss


class MultiTeacherDistillation(nn.Module):
    """
    多教师知识蒸馏
    从多个预训练教师模型中蒸馏知识到单个学生模型
    """
    
    def __init__(self, 
                 teacher_models: List,  # List of teacher models
                 student_model: nn.Module,
                 device: torch.device,
                 temperature: float = 4.0,
                 teacher_weights: Optional[List[float]] = None):
        """
        Args:
            teacher_models: 教师模型列表
            student_model: 学生模型
            device: 计算设备
            temperature: 蒸馏温度
            teacher_weights: 教师模型权重
        """
        super().__init__()
        self.teachers = teacher_models
        self.student = student_model
        self.device = device
        self.temperature = temperature
        
        # 设置教师模型权重
        if teacher_weights is None:
            self.teacher_weights = [1.0 / len(teacher_models)] * len(teacher_models)
        else:
            self.teacher_weights = teacher_weights
            
        # 冻结教师模型参数
        for teacher in self.teachers:
            for param in teacher.parameters():
                param.requires_grad = False
            teacher.eval()
            
        self.distillation_loss = KnowledgeDistillationLoss(temperature=temperature)
        
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            x: 输入数据
            labels: 真实标签   
        Returns:
            学生模型输出, 总损失
        """
        # 学生模型预测
        student_logits = self.student(x)
        
        # 教师模型预测
        teacher_logits_list = []
        with torch.no_grad():
            for teacher in self.teachers:
                teacher_output = teacher(x)
                teacher_logits_list.append(teacher_output)
        
        # 加权平均教师模型输出
        weighted_teacher_logits = torch.zeros_like(teacher_logits_list[0])
        for i, teacher_logits in enumerate(teacher_logits_list):
            weighted_teacher_logits += self.teacher_weights[i] * teacher_logits
            
        # 计算蒸馏损失
        loss = self.distillation_loss(student_logits, weighted_teacher_logits, labels)
        
        return student_logits, loss
        

class ProgressiveKnowledgeDistillation:
    """
    渐进式知识蒸馏
    逐步从简单到复杂的教师模型进行知识转移
    """
    
    def __init__(self, 
                 teacher_models: List,  # List of teacher models
                 student_model: nn.Module,
                 device: torch.device):
        self.teachers = teacher_models
        self.student = student_model
        self.device = device
        self.current_stage = 0
        
    def get_current_teachers(self) -> List:  # List of teacher models
        """
        获取当前阶段的教师模型
        """
        # 渐进式增加教师模型数量
        num_teachers = min(self.current_stage + 1, len(self.teachers))
        return self.teachers[:num_teachers]
        
    def advance_stage(self):
        """
        进入下一个蒸馏阶段
        """
        if self.current_stage < len(self.teachers) - 1:
            self.current_stage += 1
            print(f"进入蒸馏阶段 {self.current_stage + 1}, 使用 {self.current_stage + 1} 个教师模型")
            
    def create_distillation_model(self) -> MultiTeacherDistillation:
        """
        创建当前阶段的蒸馏模型
        """
        current_teachers = self.get_current_teachers()
        return MultiTeacherDistillation(
            teacher_models=current_teachers,
            student_model=self.student,
            device=self.device
        )


if __name__ == "__main__":
    # 测试知识蒸馏模块
    print("知识蒸馏模块测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试教师模型
    teacher1 = BasicModelBank(model_name='UNet', device=device)
    teacher2 = BasicModelBank(model_name='SegResNet', device=device)
    teachers = [teacher1, teacher2]
    
    # 创建测试学生模型
    student = BasicModelBank(model_name='UNet', device=device).model
    
    # 测试多教师蒸馏
    distillation = MultiTeacherDistillation(
        teacher_models=teachers,
        student_model=student,
        device=device
    )
    
    # 测试前向传播
    test_input = torch.randn(1, 4, 128, 128, 128).to(device)
    test_labels = torch.randint(0, 4, (1, 128, 128, 128)).to(device)
    
    try:
        output, loss = distillation(test_input, test_labels)
        print(f"输出形状: {output.shape}")
        print(f"损失值: {loss.item():.4f}")
        print("知识蒸馏模块测试成功！")
    except Exception as e:
        print(f"测试失败: {e}")