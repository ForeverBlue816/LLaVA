# bit_allocator.py 的完整版本
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

class BitAllocationNetwork(nn.Module):
    def __init__(self, feature_dim, min_bits=2.0, max_bits=8.0, 
                 target_bits=4.0, num_groups=8, bit_levels=[2, 4, 8]):
        super().__init__()
        
        assert feature_dim % num_groups == 0, "feature_dim must be divisible by num_groups"
        
        self.feature_dim = feature_dim
        self.num_groups = num_groups
        self.group_size = feature_dim // num_groups
        self.min_bits = min_bits
        self.max_bits = max_bits
        self.target_bits = target_bits
        
        # 改进的预测器：接受组级输入
        self.bit_predictor = nn.Sequential(
            nn.Linear(num_groups, num_groups * 2),
            nn.GELU(),
            nn.LayerNorm(num_groups * 2),
            nn.Linear(num_groups * 2, num_groups)
        )
        
        # 可学习的位嵌入
        self.bit_embeddings = nn.Parameter(
            torch.randn(len(bit_levels), self.group_size)
        )
        
        self.register_buffer('bit_levels', torch.tensor(bit_levels, dtype=torch.float32))
        self.register_buffer('temperature', torch.tensor(1.0))
    
    def allocate_bits_with_constraints(self, bit_probs):
        """根据概率分配比特，满足约束"""
        batch_size = bit_probs.size(0)
        
        # 基于概率分配比特
        total_bit_budget = self.target_bits * self.num_groups
        bit_allocation = self.min_bits + bit_probs * (self.max_bits - self.min_bits)
        
        # 归一化以满足预算
        current_total = bit_allocation.sum(dim=-1, keepdim=True)
        bit_allocation = bit_allocation * (total_bit_budget / current_total)
        
        # 确保在范围内
        bit_allocation = torch.clamp(bit_allocation, self.min_bits, self.max_bits)
        
        return bit_allocation
    
    def discretize_bits(self, bit_allocation):
        """将连续比特分配离散化到允许的级别"""
        # 找到最近的离散级别
        distances = torch.abs(
            bit_allocation.unsqueeze(-1) - self.bit_levels.view(1, 1, -1)
        )
        discrete_indices = distances.argmin(dim=-1)
        discrete_bits = self.bit_levels[discrete_indices]
        
        # Straight-through gradient
        if self.training:
            discrete_bits = discrete_bits + (bit_allocation - bit_allocation.detach())
        
        return discrete_bits
    
    def generate_bit_embeddings(self, discrete_bits):
        """生成比特嵌入"""
        batch_size = discrete_bits.size(0)
        
        # 为每个比特级别创建one-hot或soft分配
        bit_weights = []
        for level in self.bit_levels:
            weight = torch.exp(-torch.abs(discrete_bits - level))
            bit_weights.append(weight)
        
        bit_weights = torch.stack(bit_weights, dim=-1)  # [B, num_groups, num_levels]
        bit_weights = F.softmax(bit_weights, dim=-1)
        
        # 加权组合嵌入
        # bit_embeddings: [num_levels, group_size]
        # bit_weights: [B, num_groups, num_levels]
        embeddings = []
        for g in range(self.num_groups):
            group_weights = bit_weights[:, g, :]  # [B, num_levels]
            # 加权求和
            group_embedding = torch.matmul(group_weights, self.bit_embeddings)  # [B, group_size]
            embeddings.append(group_embedding)
        
        embeddings = torch.stack(embeddings, dim=1)  # [B, num_groups, group_size]
        return embeddings
    
    def forward(self, importance_scores, features=None):
        """
        importance_scores: 可以是 [B, D] 或 [B, T, D]
        """
        # 处理不同的输入维度
        if importance_scores.dim() == 3:  # [B, T, D]
            # 在时间维度上聚合
            importance_scores = importance_scores.mean(dim=1)  # [B, D]
        
        batch_size = importance_scores.size(0)
        
        # 1. 计算组级重要性
        importance_reshaped = importance_scores.view(batch_size, self.num_groups, -1)
        group_importance = importance_reshaped.mean(dim=-1)  # [B, num_groups]
        
        # 2. 预测比特分配
        bit_logits = self.bit_predictor(group_importance)
        
        # 3. Gumbel-Softmax
        if self.training:
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(bit_logits) + 1e-8) + 1e-8)
            bit_logits = bit_logits + gumbel_noise
        
        bit_probs = F.softmax(bit_logits / self.temperature, dim=-1)
        
        # 4. 分配比特（带约束）
        bit_assignment = self.allocate_bits_with_constraints(bit_probs)
        
        # 5. 离散化
        discrete_bits = self.discretize_bits(bit_assignment)
        
        # 6. 生成比特嵌入
        bit_embeddings = self.generate_bit_embeddings(discrete_bits)
        
        # 7. 生成group_indices（重要！）
        group_indices = torch.arange(self.feature_dim, device=importance_scores.device)
        group_indices = group_indices // self.group_size
        group_indices = group_indices.unsqueeze(0).expand(batch_size, -1)
        
        # 8. 计算损失
        bitrate_loss = F.mse_loss(
            discrete_bits.mean(), 
            torch.tensor(self.target_bits, device=discrete_bits.device)
        )
        
        entropy = -(bit_probs * torch.log(bit_probs + 1e-8)).sum(dim=-1).mean()
        diversity_bonus = entropy
        
        return {
            'bit_assignment': discrete_bits,
            'group_indices': group_indices,  # 添加这个！
            'bit_embeddings': bit_embeddings,
            'bitrate_loss': bitrate_loss,
            'diversity_bonus': diversity_bonus,
            'bit_probs': bit_probs
        }