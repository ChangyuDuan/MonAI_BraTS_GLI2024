import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional
import math
from monai.networks.nets import UNet, SegResNet, UNETR


class CrossAttentionFusion(nn.Module):
    """
    跨模型交叉注意力融合模块
    """
    
    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        self.query_proj = nn.Linear(feature_dim, feature_dim)
        self.key_proj = nn.Linear(feature_dim, feature_dim)
        self.value_proj = nn.Linear(feature_dim, feature_dim)
        self.output_proj = nn.Linear(feature_dim, feature_dim)
        
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(feature_dim)
        
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: 来自不同模型的特征列表
        Returns:
            融合后的特征
        """
        batch_size = features_list[0].size(0)
        seq_len = features_list[0].numel() // (batch_size * self.feature_dim)
        
        # 将所有特征拼接
        all_features = torch.stack(features_list, dim=1)  # [B, N_models, ...]
        all_features = all_features.view(batch_size, len(features_list), -1, self.feature_dim)
        
        # 计算注意力权重
        queries = self.query_proj(all_features)
        keys = self.key_proj(all_features)
        values = self.value_proj(all_features)
        
        # 多头注意力
        queries = queries.view(batch_size, len(features_list), -1, self.num_heads, self.head_dim)
        keys = keys.view(batch_size, len(features_list), -1, self.num_heads, self.head_dim)
        values = values.view(batch_size, len(features_list), -1, self.num_heads, self.head_dim)
        
        # 计算注意力分数
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-2)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力权重
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.view(batch_size, len(features_list), -1, self.feature_dim)
        
        # 输出投影
        output = self.output_proj(attended_values)
        
        # 残差连接和层归一化
        output = self.layer_norm(output + all_features)
        
        # 平均融合
        fused_features = torch.mean(output, dim=1)
        
        return fused_features


class ChannelAttentionFusion(nn.Module):
    """
    通道注意力融合模块
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super().__init__()
        self.in_channels = in_channels
        self.reduction_ratio = reduction_ratio
        
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: 来自不同模型的特征列表
        Returns:
            融合后的特征
        """
        # 计算每个特征的通道注意力权重
        attention_weights = []
        
        for features in features_list:
            # 全局平均池化和最大池化
            avg_pool = self.global_avg_pool(features).view(features.size(0), -1)
            max_pool = self.global_max_pool(features).view(features.size(0), -1)
            
            # 计算注意力权重
            avg_weight = self.fc(avg_pool)
            max_weight = self.fc(max_pool)
            
            # 组合权重
            weight = self.sigmoid(avg_weight + max_weight)
            attention_weights.append(weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
        
        # 应用注意力权重并融合
        weighted_features = []
        for i, features in enumerate(features_list):
            weighted = features * attention_weights[i]
            weighted_features.append(weighted)
        
        # 加权平均融合
        fused_features = torch.stack(weighted_features, dim=0).mean(dim=0)
        
        return fused_features


class SpatialAttentionFusion(nn.Module):
    """
    空间注意力融合模块
    """
    
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv3d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: 来自不同模型的特征列表
        Returns:
            融合后的特征
        """
        # 计算每个特征的空间注意力权重
        attention_maps = []
        
        for features in features_list:
            # 计算通道维度的平均值和最大值
            avg_pool = torch.mean(features, dim=1, keepdim=True)
            max_pool, _ = torch.max(features, dim=1, keepdim=True)
            
            # 拼接平均值和最大值
            concat = torch.cat([avg_pool, max_pool], dim=1)
            
            # 计算空间注意力权重
            attention_map = self.sigmoid(self.conv(concat))
            attention_maps.append(attention_map)
        
        # 应用空间注意力权重并融合
        weighted_features = []
        for i, features in enumerate(features_list):
            weighted = features * attention_maps[i]
            weighted_features.append(weighted)
        
        # 加权平均融合
        fused_features = torch.stack(weighted_features, dim=0).mean(dim=0)
        
        return fused_features


class AdaptiveFusionGate(nn.Module):
    """
    自适应融合门控机制
    """
    
    def __init__(self, feature_dim: int, num_models: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_models = num_models
        
        # 门控网络
        self.gate_network = nn.Sequential(
            nn.Linear(feature_dim * num_models, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, num_models),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, features_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            features_list: 来自不同模型的特征列表
        Returns:
            融合后的特征
        """
        batch_size = features_list[0].size(0)
        
        # 将所有特征展平并拼接
        flattened_features = []
        for features in features_list:
            flattened = features.view(batch_size, -1)
            flattened_features.append(flattened)
        
        concat_features = torch.cat(flattened_features, dim=-1)
        
        # 计算门控权重
        gate_weights = self.gate_network(concat_features)  # [B, num_models]
        
        # 应用门控权重
        weighted_features = []
        for i, features in enumerate(features_list):
            weight = gate_weights[:, i].view(-1, 1, 1, 1, 1)
            weighted = features * weight
            weighted_features.append(weighted)
        
        # 加权求和
        fused_features = torch.stack(weighted_features, dim=0).sum(dim=0)
        
        return fused_features


class FusionNetworkArchitecture(nn.Module):
    """
    专门的融合网络架构
    整合多种融合机制
    """
    
    def __init__(self, 
                 base_models: List[nn.Module],
                 fusion_channels: List[int] = [64, 128, 256, 512],
                 num_classes: int = 4,
                 fusion_type: str = 'cross_attention'):
        super().__init__()
        self.base_models = nn.ModuleList(base_models)
        self.num_models = len(base_models)
        self.fusion_channels = fusion_channels
        self.num_classes = num_classes
        self.fusion_type = fusion_type
        
        # 冻结基础模型参数
        for model in self.base_models:
            for param in model.parameters():
                param.requires_grad = False
        
        # 多级融合模块
        self.cross_attention_fusions = nn.ModuleList([
            CrossAttentionFusion(ch) for ch in fusion_channels
        ])
        
        self.channel_attention_fusions = nn.ModuleList([
            ChannelAttentionFusion(ch) for ch in fusion_channels
        ])
        
        self.spatial_attention_fusions = nn.ModuleList([
            SpatialAttentionFusion() for _ in fusion_channels
        ])
        
        self.adaptive_gates = nn.ModuleList([
            AdaptiveFusionGate(ch, self.num_models) for ch in fusion_channels
        ])
        
        # 融合权重学习
        self.fusion_weights = nn.Parameter(torch.ones(4, len(fusion_channels)))
        
        # 最终解码器
        self.decoder = self._build_decoder()
        
    def _build_decoder(self) -> nn.Module:
        """
        构建统一解码器
        """
        return nn.Sequential(
            nn.Conv3d(self.fusion_channels[-1], 256, 3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(256, 128, 2, stride=2),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose3d(128, 64, 2, stride=2),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(64, self.num_classes, 1)
        )
        
    def extract_multi_level_features(self, x: torch.Tensor) -> List[List[torch.Tensor]]:
        """
        从所有基础模型中提取多级特征
        Args:
            x: 输入数据
        Returns:
            多级特征列表 [level][model]
        """
        all_features = []
        
        for i, model in enumerate(self.base_models):
            # 获取模型输出
            with torch.no_grad():
                output = model(x)  # 形状: [batch, 4, H, W, D]
            
            # 为每个融合层级创建特征
            # 通过不同的池化操作创建多尺度特征
            model_features = []
            
            for level, target_channels in enumerate(self.fusion_channels):
                # 使用自适应池化调整空间尺寸
                pooled_output = F.adaptive_avg_pool3d(output, 
                                                     (output.size(2) // (2**level), 
                                                      output.size(3) // (2**level), 
                                                      output.size(4) // (2**level)))
                
                # 使用1x1x1卷积调整通道数
                if not hasattr(self, f'channel_adapters_{i}'):
                    # 动态创建通道适配器
                    setattr(self, f'channel_adapters_{i}', 
                           nn.ModuleList([
                               nn.Conv3d(4, ch, 1).to(x.device) for ch in self.fusion_channels
                           ]))
                
                channel_adapters = getattr(self, f'channel_adapters_{i}')
                adapted_feature = channel_adapters[level](pooled_output)
                model_features.append(adapted_feature)
            
            all_features.append(model_features)
        
        # 重新组织为 [level][model] 格式
        level_features = []
        for level in range(len(self.fusion_channels)):
            level_feature_list = []
            for model_idx in range(len(all_features)):
                if level < len(all_features[model_idx]):
                    feature = all_features[model_idx][level]
                    level_feature_list.append(feature)
            
            if level_feature_list:
                level_features.append(level_feature_list)
        
        return level_features
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        Args:
            x: 输入数据   
        Returns:
            融合预测结果
        """
        # 提取多级特征
        level_features = self.extract_multi_level_features(x)
        
        # 多级融合
        fused_features = []
        
        for level, features_list in enumerate(level_features):
            if level >= len(self.fusion_channels):
                break
                
            # 应用不同的融合策略
            cross_fused = self.cross_attention_fusions[level](features_list)
            channel_fused = self.channel_attention_fusions[level](features_list)
            spatial_fused = self.spatial_attention_fusions[level](features_list)
            gate_fused = self.adaptive_gates[level](features_list)
            
            # 加权组合不同融合策略
            weights = F.softmax(self.fusion_weights[:, level], dim=0)
            level_fused = (weights[0] * cross_fused + 
                          weights[1] * channel_fused + 
                          weights[2] * spatial_fused + 
                          weights[3] * gate_fused)
            
            fused_features.append(level_fused)
        
        # 使用最高级特征进行最终预测
        if fused_features:
            final_features = fused_features[-1]
            output = self.decoder(final_features)
        else:
            # 如果没有提取到特征，使用第一个模型的输出
            output = self.base_models[0](x)
        
        return output


if __name__ == "__main__":
    # 测试融合网络架构
    print("融合网络架构测试...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试基础模型
    base_models = [
        UNet(spatial_dims=3, in_channels=4, out_channels=4, channels=(32, 64), strides=(2,)),
        UNet(spatial_dims=3, in_channels=4, out_channels=4, channels=(32, 64), strides=(2,))
    ]
    
    # 创建融合网络
    fusion_net = FusionNetworkArchitecture(
        base_models=base_models,
        fusion_channels=[32, 64],
        num_classes=4
    ).to(device)
    
    # 测试前向传播
    test_input = torch.randn(1, 4, 64, 64, 64).to(device)
    
    try:
        output = fusion_net(test_input)
        print(f"输出形状: {output.shape}")
        print("融合网络架构测试成功！")
    except Exception as e:
        print(f"测试失败: {e}")