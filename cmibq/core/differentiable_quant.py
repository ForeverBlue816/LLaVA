import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class StraightThroughEstimator(torch.autograd.Function):
    """
    Straight-through estimator for gradient flow
    """
    @staticmethod
    def forward(ctx, input, scale, zero_point, num_bits):
        # 修复1：处理num_bits的类型问题
        if isinstance(num_bits, torch.Tensor):
            num_bits = int(num_bits.item())
        else:
            num_bits = int(num_bits)
            
        n_levels = 2 ** num_bits  # 现在安全了
        q_min, q_max = 0, n_levels - 1
        
        # 修复2：防止scale为0
        eps = 1e-8
        scale = scale.clamp(min=eps)
        
        # Quantize
        input_scaled = input / scale + zero_point
        input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
        
        # Dequantize
        output = (input_quantized - zero_point) * scale
        
        # Save for backward
        ctx.save_for_backward(input)
        ctx.scale = scale
        ctx.zero_point = zero_point
        ctx.q_min = q_min
        ctx.q_max = q_max
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        
        # Straight-through gradient
        grad_input = grad_output.clone()
        
        # Zero gradient outside valid range
        input_scaled = input / ctx.scale + ctx.zero_point
        mask = (input_scaled >= ctx.q_min) & (input_scaled <= ctx.q_max)
        grad_input = grad_input * mask.float()
        
        return grad_input, None, None, None


class DifferentiableQuantizer(nn.Module):
    """
    Differentiable quantizer with learnable parameters
    """
    def __init__(
        self,
        feature_dim: int,
        default_bits: int = 8,
        per_channel: bool = True,
        symmetric: bool = False,
        per_sample: bool = False,  # 新增：是否支持样本级自适应
        bit_levels: list = [2, 4, 8],  # 新增：支持的离散比特级别
        target_budget: Optional[float] = None  # 新增：目标比特预算
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.default_bits = default_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.eps = 1e-8  # 添加eps常量
        self.per_sample = per_sample  # 是否样本级自适应
        self.bit_levels = bit_levels
        self.target_budget = target_budget if target_budget is not None else default_bits
        
        # Learnable quantization parameters
        if per_channel:
            self.scale = nn.Parameter(torch.ones(1, 1, feature_dim))
            self.zero_point = nn.Parameter(torch.zeros(1, 1, feature_dim))
        else:
            self.scale = nn.Parameter(torch.ones(1))
            self.zero_point = nn.Parameter(torch.zeros(1))
        
        # Calibration stats
        self.register_buffer('running_min', torch.zeros_like(self.scale))
        self.register_buffer('running_max', torch.zeros_like(self.scale))
        self.register_buffer('calibration_counter', torch.tensor(0))
    
    def forward(
        self,
        x: torch.Tensor,
        bit_assignment: Optional[torch.Tensor] = None,
        group_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Calibrate if needed
        if self.training and self.calibration_counter < 100:
            self.calibrate(x)
        
        if bit_assignment is not None and group_indices is not None:
            # 先离散化比特分配并满足预算约束
            discrete_bits = self.discretize_with_budget(
                bit_assignment, 
                self.target_budget, 
                self.bit_levels
            )
            
            # 选择量化方式
            if self.per_sample:
                return self.adaptive_quantize_per_sample(x, discrete_bits, group_indices)
            else:
                return self.adaptive_quantize(x, discrete_bits, group_indices)
        else:
            # Uniform quantization
            if self.training:
                # 修复：直接传int而不是tensor
                return StraightThroughEstimator.apply(
                    x, self.scale, self.zero_point, self.default_bits  # 不要torch.tensor()
                )
            else:
                n_levels = 2 ** self.default_bits
                q_min, q_max = 0, n_levels - 1
                scale_safe = self.scale.clamp(min=self.eps)  # 防止除零
                input_scaled = x / scale_safe + self.zero_point
                input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
                return (input_quantized - self.zero_point) * scale_safe
    
    def calibrate(self, x: torch.Tensor):
        """Update calibration statistics"""
        with torch.no_grad():
            if self.per_channel:
                x_min = x.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
                x_max = x.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
            else:
                x_min = x.min()
                x_max = x.max()
            
            if self.calibration_counter == 0:
                self.running_min.copy_(x_min)
                self.running_max.copy_(x_max)
            else:
                momentum = 0.9
                self.running_min.mul_(momentum).add_(x_min, alpha=1-momentum)
                self.running_max.mul_(momentum).add_(x_max, alpha=1-momentum)
            
            self.calibration_counter += 1
            
            # Update scale and zero_point - 修复3个问题
            if self.symmetric:
                abs_max = torch.max(self.running_max.abs(), self.running_min.abs())
                abs_max = abs_max.clamp(min=self.eps)  # 防止为0
                # 修复：使用正确的对称量化公式
                self.scale.data = abs_max / (2 ** (self.default_bits - 1) - 1)
                self.scale.data.clamp_(min=self.eps)  # 再次确保不为0
                self.zero_point.data.zero_()
            else:
                range_val = self.running_max - self.running_min
                range_val = range_val.clamp(min=self.eps)  # 防止为0
                self.scale.data = range_val / (2 ** self.default_bits - 1)
                self.scale.data.clamp_(min=self.eps)  # 确保不为0
                self.zero_point.data = -self.running_min / self.scale.data
    
    def adaptive_quantize(
        self,
        x: torch.Tensor,
        bit_assignment: torch.Tensor,
        group_indices: torch.Tensor
    ) -> torch.Tensor:
        """Apply mixed-precision quantization (batch-level)"""
        batch_size, seq_len, feature_dim = x.shape
        num_groups = bit_assignment.size(1)
        x_quantized = torch.zeros_like(x)
        
        for g in range(num_groups):
            # Get group mask and features
            group_mask = (group_indices == g).unsqueeze(1).expand_as(x)
            group_features = x * group_mask.float()
            
            # Get bits for this group - 使用离散化后的比特
            group_bits = int(bit_assignment[:, g].mean().item())
            
            # Quantize group
            if self.training:
                group_quantized = StraightThroughEstimator.apply(
                    group_features,
                    self.scale,
                    self.zero_point,
                    group_bits  # 传入int
                )
            else:
                n_levels = 2 ** group_bits
                q_min, q_max = 0, n_levels - 1
                scale_safe = self.scale.clamp(min=self.eps)  # 防止除零
                input_scaled = group_features / scale_safe + self.zero_point
                input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
                group_quantized = (input_quantized - self.zero_point) * scale_safe
            
            x_quantized = x_quantized + group_quantized * group_mask.float()
        
        return x_quantized
    
    def adaptive_quantize_per_sample(
        self,
        x: torch.Tensor,
        bit_assignment: torch.Tensor,
        group_indices: torch.Tensor
    ) -> torch.Tensor:
        """Apply mixed-precision quantization (per-sample level)"""
        batch_size, seq_len, feature_dim = x.shape
        num_groups = bit_assignment.size(1)
        x_quantized = torch.zeros_like(x)
        
        for b in range(batch_size):
            for g in range(num_groups):
                # Get group mask for this sample
                if group_indices.dim() == 2:  # [B, D]
                    group_mask = (group_indices[b] == g).unsqueeze(0).expand(seq_len, -1)
                else:  # [B, T, D]
                    group_mask = (group_indices[b] == g)
                
                # Get features for this group and sample
                sample_features = x[b] * group_mask.float()
                
                # Get bits for this specific sample and group
                sample_bits = int(bit_assignment[b, g].item())
                
                # Quantize
                if self.training:
                    # Expand dims for STE
                    sample_features_expanded = sample_features.unsqueeze(0)  # [1, T, D]
                    
                    # Get appropriate scale/zero_point slice if per_channel
                    if self.per_channel:
                        # Select channels belonging to this group
                        group_channels = group_mask[0] if seq_len > 0 else group_mask
                        scale_group = self.scale[:, :, group_channels]
                        zero_point_group = self.zero_point[:, :, group_channels]
                    else:
                        scale_group = self.scale
                        zero_point_group = self.zero_point
                    
                    group_quantized = StraightThroughEstimator.apply(
                        sample_features_expanded,
                        scale_group,
                        zero_point_group,
                        sample_bits
                    )
                    x_quantized[b] += group_quantized.squeeze(0) * group_mask.float()
                else:
                    # Inference mode
                    n_levels = 2 ** sample_bits
                    q_min, q_max = 0, n_levels - 1
                    
                    if self.per_channel:
                        group_channels = group_mask[0] if seq_len > 0 else group_mask
                        scale_safe = self.scale[:, :, group_channels].squeeze(0).clamp(min=self.eps)
                        zero_point_use = self.zero_point[:, :, group_channels].squeeze(0)
                    else:
                        scale_safe = self.scale.clamp(min=self.eps)
                        zero_point_use = self.zero_point
                    
                    input_scaled = sample_features / scale_safe + zero_point_use
                    input_quantized = torch.clamp(torch.round(input_scaled), q_min, q_max)
                    group_quantized = (input_quantized - zero_point_use) * scale_safe
                    
                    x_quantized[b] += group_quantized * group_mask.float()
        
        return x_quantized
    
    def discretize_with_budget(
        self, 
        bit_assignment: torch.Tensor,
        target_budget: float,
        bit_levels: list
    ) -> torch.Tensor:
        """
        Discretize bit assignment to allowed levels and enforce budget constraint
        
        Args:
            bit_assignment: [B, num_groups] continuous bit assignments
            target_budget: target average bits per group
            bit_levels: allowed discrete bit levels (e.g., [2, 4, 8])
        
        Returns:
            discrete_bits: [B, num_groups] discretized bit assignments
        """
        batch_size, num_groups = bit_assignment.shape
        bit_levels_tensor = torch.tensor(bit_levels, device=bit_assignment.device, dtype=torch.float32)
        
        # Step 1: Find nearest discrete level for each assignment
        distances = torch.abs(bit_assignment.unsqueeze(-1) - bit_levels_tensor.view(1, 1, -1))
        discrete_bits = bit_levels_tensor[distances.argmin(dim=-1)]
        
        # Step 2: Adjust to meet budget constraint per sample
        for b in range(batch_size):
            current_budget = discrete_bits[b].mean()
            iterations = 0
            max_iterations = num_groups * 2  # Prevent infinite loops
            
            while abs(current_budget - target_budget) > 0.1 and iterations < max_iterations:
                iterations += 1
                
                if current_budget > target_budget:
                    # Need to reduce bits
                    # Find groups that can be reduced
                    reducible = discrete_bits[b] > min(bit_levels)
                    if not reducible.any():
                        break
                    
                    # Prioritize reducing groups with lower importance (higher original assignment)
                    # since higher bits were likely assigned to more important groups
                    reducible_indices = torch.where(reducible)[0]
                    
                    # Select the group with highest current bits to reduce
                    max_bit_idx = reducible_indices[discrete_bits[b, reducible_indices].argmax()]
                    
                    # Reduce by one level
                    current_level_idx = (discrete_bits[b, max_bit_idx] == bit_levels_tensor).nonzero()[0, 0]
                    if current_level_idx > 0:
                        discrete_bits[b, max_bit_idx] = bit_levels_tensor[current_level_idx - 1]
                    
                else:
                    # Need to increase bits
                    # Find groups that can be increased
                    increasable = discrete_bits[b] < max(bit_levels)
                    if not increasable.any():
                        break
                    
                    # Select the group with lowest current bits to increase
                    increasable_indices = torch.where(increasable)[0]
                    min_bit_idx = increasable_indices[discrete_bits[b, increasable_indices].argmin()]
                    
                    # Increase by one level
                    current_level_idx = (discrete_bits[b, min_bit_idx] == bit_levels_tensor).nonzero()[0, 0]
                    if current_level_idx < len(bit_levels) - 1:
                        discrete_bits[b, min_bit_idx] = bit_levels_tensor[current_level_idx + 1]
                
                current_budget = discrete_bits[b].mean()
        
        return discrete_bits