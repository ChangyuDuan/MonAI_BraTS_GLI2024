import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Callable
import numpy as np
from collections import OrderedDict
import copy


class MixedOperation(nn.Module):
    """
    混合操作：包含多种候选操作的可微分搜索单元
    """
    
    def __init__(self, operations: List[nn.Module]):
        super().__init__()
        self.operations = nn.ModuleList(operations)
        self.num_ops = len(operations)
        
    def forward(self, x: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入特征
            weights: 操作权重 [num_ops]
        Returns:
            加权操作结果
        """
        # 应用softmax确保权重和为1
        weights = F.softmax(weights, dim=0)
        
        # 加权组合所有操作的输出
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        return output


class SearchableConvBlock(nn.Module):
    """
    可搜索的卷积块
    """
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        
        # 定义候选操作
        operations = [
            # 标准卷积
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, stride, 1, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ),
            # 深度可分离卷积
            nn.Sequential(
                nn.Conv3d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.Conv3d(in_channels, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ),
            # 扩张卷积
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 3, stride, 2, dilation=2, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ),
            # 1x1卷积
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
            ),
            # 跳跃连接（如果维度匹配）
            nn.Identity() if in_channels == out_channels and stride == 1 else 
            nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        ]
        
        self.mixed_op = MixedOperation(operations)
        
    def forward(self, x: torch.Tensor, arch_weights: torch.Tensor) -> torch.Tensor:
        return self.mixed_op(x, arch_weights)


class SearchableAttentionBlock(nn.Module):
    """
    可搜索的注意力块
    """
    
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        
        # 定义候选注意力机制
        operations = [
            # 通道注意力
            self._create_channel_attention(channels),
            # 空间注意力
            self._create_spatial_attention(),
            # 自注意力
            self._create_self_attention(channels),
            # 无注意力（恒等映射）
            nn.Identity()
        ]
        
        self.mixed_op = MixedOperation(operations)
        
    def _create_channel_attention(self, channels: int) -> nn.Module:
        return nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, channels // 16, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels // 16, channels, 1),
            nn.Sigmoid()
        )
        
    def _create_spatial_attention(self) -> nn.Module:
        return nn.Sequential(
            nn.Conv3d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )
        
    def _create_self_attention(self, channels: int) -> nn.Module:
        return nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=8,
            batch_first=True
        )
        
    def forward(self, x: torch.Tensor, arch_weights: torch.Tensor) -> torch.Tensor:
        return self.mixed_op(x, arch_weights)


class SuperNet(nn.Module):
    """
    超网络：包含所有可能架构的搜索空间
    """
    
    def __init__(self, 
                 in_channels: int = 4,
                 num_classes: int = 4,
                 base_channels: int = 32,
                 num_layers: int = 4):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # 构建搜索空间
        self.stem = nn.Conv3d(in_channels, base_channels, 3, 1, 1, bias=False)
        
        # 编码器层
        self.encoder_layers = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        
        current_channels = base_channels
        for i in range(num_layers):
            out_channels = current_channels * 2
            stride = 2 if i > 0 else 1
            
            # 卷积块
            conv_block = SearchableConvBlock(current_channels, out_channels, stride)
            self.encoder_layers.append(conv_block)
            
            # 注意力块
            attention_block = SearchableAttentionBlock(out_channels)
            self.attention_layers.append(attention_block)
            
            current_channels = out_channels
        
        # 解码器
        self.decoder = self._build_decoder(current_channels)
        
        # 架构参数
        self.arch_parameters = self._initialize_arch_parameters()
        
    def _build_decoder(self, in_channels: int) -> nn.Module:
        return nn.Sequential(
            nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2),
            nn.BatchNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(in_channels // 2, in_channels // 4, 2, stride=2),
            nn.BatchNorm3d(in_channels // 4),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(in_channels // 4, self.num_classes, 1)
        )
        
    def _initialize_arch_parameters(self) -> nn.ParameterDict:
        """
        初始化架构参数
        """
        arch_params = nn.ParameterDict()
        
        # 为每个搜索块初始化架构参数
        for i in range(self.num_layers):
            # 卷积操作权重
            conv_weights = torch.randn(len(self.encoder_layers[i].mixed_op.operations))
            arch_params[f'conv_layer_{i}'] = nn.Parameter(conv_weights)
            
            # 注意力操作权重
            attn_weights = torch.randn(len(self.attention_layers[i].mixed_op.operations))
            arch_params[f'attn_layer_{i}'] = nn.Parameter(attn_weights)
            
        return arch_params
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Stem
        x = self.stem(x)
        
        # 编码器
        for i in range(self.num_layers):
            # 卷积块
            conv_weights = self.arch_parameters[f'conv_layer_{i}']
            x = self.encoder_layers[i](x, conv_weights)
            
            # 注意力块
            attn_weights = self.arch_parameters[f'attn_layer_{i}']
            attention_out = self.attention_layers[i](x, attn_weights)
            
            # 残差连接
            if attention_out.shape == x.shape:
                x = x + attention_out
            else:
                x = attention_out
        
        # 解码器
        output = self.decoder(x)
        
        return output
        
    def get_arch_parameters(self) -> List[nn.Parameter]:
        """
        获取架构参数
        """
        return list(self.arch_parameters.parameters())
        
    def get_model_parameters(self) -> List[nn.Parameter]:
        """
        获取模型参数（非架构参数）
        """
        model_params = []
        for name, param in self.named_parameters():
            if not name.startswith('arch_parameters'):
                model_params.append(param)
        return model_params


class DARTSSearcher:
    """
    DARTS (Differentiable Architecture Search) 搜索器
    """
    
    def __init__(self, 
                 supernet: SuperNet,
                 device: torch.device,
                 arch_lr: float = 3e-4,
                 model_lr: float = 1e-3):
        self.supernet = supernet.to(device)
        self.device = device
        
        # 分别优化架构参数和模型参数
        self.arch_optimizer = torch.optim.Adam(
            self.supernet.get_arch_parameters(),
            lr=arch_lr,
            betas=(0.5, 0.999),
            weight_decay=1e-3
        )
        
        self.model_optimizer = torch.optim.SGD(
            self.supernet.get_model_parameters(),
            lr=model_lr,
            momentum=0.9,
            weight_decay=3e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
    def search_step(self, 
                   train_data: torch.Tensor,
                   train_labels: torch.Tensor,
                   val_data: torch.Tensor,
                   val_labels: torch.Tensor) -> Tuple[float, float]:
        """
        执行一步DARTS搜索
        
        Args:
            train_data: 训练数据
            train_labels: 训练标签
            val_data: 验证数据
            val_labels: 验证标签
            
        Returns:
            训练损失, 验证损失
        """
        # 第一步：更新模型参数
        self.model_optimizer.zero_grad()
        train_pred = self.supernet(train_data)
        train_loss = self.criterion(train_pred, train_labels)
        train_loss.backward()
        self.model_optimizer.step()
        
        # 第二步：更新架构参数
        self.arch_optimizer.zero_grad()
        val_pred = self.supernet(val_data)
        val_loss = self.criterion(val_pred, val_labels)
        val_loss.backward()
        self.arch_optimizer.step()
        
        return train_loss.item(), val_loss.item()
        
    def derive_architecture(self) -> Dict[str, int]:
        """
        从搜索结果中导出最优架构
        
        Returns:
            最优架构配置
        """
        architecture = {}
        
        with torch.no_grad():
            for name, param in self.supernet.arch_parameters.items():
                # 选择权重最大的操作
                best_op_idx = torch.argmax(F.softmax(param, dim=0)).item()
                architecture[name] = best_op_idx
                
        return architecture
        
    def export_searched_model(self, architecture: Dict[str, int]) -> nn.Module:
        """
        根据搜索结果导出最终模型
        
        Args:
            architecture: 架构配置
            
        Returns:
            最终优化的模型
        """
        # 这里需要根据架构配置构建最终模型
        # 简化实现：返回当前超网络的副本
        final_model = copy.deepcopy(self.supernet)
        
        # 固定架构参数
        for name, op_idx in architecture.items():
            if name in final_model.arch_parameters:
                # 创建one-hot权重
                num_ops = final_model.arch_parameters[name].size(0)
                one_hot = torch.zeros(num_ops)
                one_hot[op_idx] = 1.0
                final_model.arch_parameters[name].data = one_hot
                final_model.arch_parameters[name].requires_grad = False
        
        return final_model


class ProgressiveNAS:
    """
    渐进式神经架构搜索
    逐步增加搜索复杂度
    """
    
    def __init__(self, 
                 device: torch.device,
                 max_layers: int = 8,
                 start_layers: int = 2):
        self.device = device
        self.max_layers = max_layers
        self.current_layers = start_layers
        self.search_history = []
        
    def create_supernet(self, num_layers: int) -> SuperNet:
        """
        创建指定层数的超网络
        """
        return SuperNet(
            in_channels=4,
            num_classes=4,
            base_channels=32,
            num_layers=num_layers
        )
        
    def progressive_search(self, 
                          train_loader,
                          val_loader,
                          epochs_per_stage: int = 50) -> Dict[str, int]:
        """
        执行渐进式搜索
        
        Args:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
            epochs_per_stage: 每个阶段的训练轮数
            
        Returns:
            最终架构配置
        """
        best_architecture = None
        best_val_loss = float('inf')
        
        for num_layers in range(self.current_layers, self.max_layers + 1):
            print(f"\n开始搜索 {num_layers} 层架构...")
            
            # 创建当前阶段的超网络
            supernet = self.create_supernet(num_layers)
            searcher = DARTSSearcher(supernet, self.device)
            
            # 搜索当前阶段
            stage_best_loss = float('inf')
            
            for epoch in range(epochs_per_stage):
                epoch_train_losses = []
                epoch_val_losses = []
                
                for train_batch, val_batch in zip(train_loader, val_loader):
                    train_data, train_labels = train_batch
                    val_data, val_labels = val_batch
                    
                    train_data = train_data.to(self.device)
                    train_labels = train_labels.to(self.device)
                    val_data = val_data.to(self.device)
                    val_labels = val_labels.to(self.device)
                    
                    train_loss, val_loss = searcher.search_step(
                        train_data, train_labels, val_data, val_labels
                    )
                    
                    epoch_train_losses.append(train_loss)
                    epoch_val_losses.append(val_loss)
                
                avg_val_loss = np.mean(epoch_val_losses)
                if avg_val_loss < stage_best_loss:
                    stage_best_loss = avg_val_loss
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch + 1}: Train Loss = {np.mean(epoch_train_losses):.4f}, "
                          f"Val Loss = {avg_val_loss:.4f}")
            
            # 导出当前阶段的最佳架构
            stage_architecture = searcher.derive_architecture()
            self.search_history.append({
                'num_layers': num_layers,
                'architecture': stage_architecture,
                'val_loss': stage_best_loss
            })
            
            # 更新全局最佳架构
            if stage_best_loss < best_val_loss:
                best_val_loss = stage_best_loss
                best_architecture = stage_architecture
                
            print(f"{num_layers} 层架构搜索完成，验证损失: {stage_best_loss:.4f}")
        
        print(f"\n渐进式搜索完成！最佳验证损失: {best_val_loss:.4f}")
        return best_architecture


if __name__ == "__main__":
    # 测试NAS模块
    print("神经架构搜索模块测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试超网络
    supernet = SuperNet(
        in_channels=4,
        num_classes=4,
        base_channels=16,  # 减小以便测试
        num_layers=2
    )
    
    # 创建DARTS搜索器
    searcher = DARTSSearcher(supernet, device)
    
    # 测试搜索步骤
    test_train_data = torch.randn(2, 4, 32, 32, 32).to(device)
    test_train_labels = torch.randint(0, 4, (2, 32, 32, 32)).to(device)
    test_val_data = torch.randn(2, 4, 32, 32, 32).to(device)
    test_val_labels = torch.randint(0, 4, (2, 32, 32, 32)).to(device)
    
    try:
        train_loss, val_loss = searcher.search_step(
            test_train_data, test_train_labels,
            test_val_data, test_val_labels
        )
        print(f"搜索步骤成功: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        
        # 测试架构导出
        architecture = searcher.derive_architecture()
        print(f"导出架构: {architecture}")
        
        print("神经架构搜索模块测试成功！")
    except Exception as e:
        print(f"测试失败: {e}")