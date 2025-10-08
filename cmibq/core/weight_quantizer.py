# cmibq/core/weight_quantizer.py
"""
权重量化模块 - 完整实现
支持per-channel和per-group的混合精度权重量化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Tuple, List
import math


class QuantizedLinear(nn.Module):
    """
    量化的Linear层，支持混合精度权重量化
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        num_bits: int = 4,
        per_channel: bool = True,
        symmetric: bool = True,
        num_groups: int = 1,
        use_adaptive: bool = False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_bits = num_bits
        self.per_channel = per_channel
        self.symmetric = symmetric
        self.num_groups = num_groups
        self.use_adaptive = use_adaptive
        
        # 原始权重（FP16/FP32）
        self.register_buffer('weight_fp', torch.randn(out_features, in_features))
        
        if bias:
            self.register_buffer('bias', torch.zeros(out_features))
        else:
            self.register_buffer('bias', None)
        
        # 创建增强的量化器（支持细粒度和离群值处理）
        if granularity == 'per_channel':
            quantizer_dim = out_features
        elif granularity == 'per_group':
            quantizer_dim = in_features  
        else:  # per_tensor
            quantizer_dim = 1
        
        self.weight_quantizer = DifferentiableQuantizer(
            feature_dim=quantizer_dim,
            default_bits=num_bits,
            per_channel=(granularity == 'per_channel'),
            symmetric=symmetric,
            granularity=granularity,
            group_size=group_size,
            use_outlier_clipping=use_outlier_clipping,
            clip_percentile=clip_percentile,
            learnable_clip=learnable_clip
        )
        
        # 如果使用自适应量化，创建比特分配
        if use_adaptive:
            self.register_buffer('bit_assignment', torch.full((num_groups,), float(num_bits)))
        
        # 量化的权重
        self.register_buffer('weight_quantized', None)
        
        # 离群值统计
        self.register_buffer('num_outliers', torch.tensor(0))
        self.register_buffer('total_elements', torch.tensor(0))
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.kaiming_uniform_(self.weight_fp, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_fp)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def quantize_weight(self, bit_assignment: Optional[torch.Tensor] = None):
        """
        量化权重（支持细粒度和离群值处理）
        
        Args:
            bit_assignment: [num_groups] 每组的比特分配
        """
        if bit_assignment is not None and self.use_adaptive:
            self.bit_assignment = bit_assignment
        
        weight = self.weight_fp
        
        # 根据粒度处理权重维度
        if self.granularity == 'per_channel':
            # 权重shape: [out_features, in_features]
            # per_channel沿out_features维度量化
            weight_for_quant = weight.unsqueeze(0).transpose(1, 2)  # [1, in_features, out_features]
            
            # 应用量化器
            if self.use_adaptive and bit_assignment is not None:
                # 混合精度量化
                weight_quantized = self._mixed_precision_quantize_with_outliers(
                    weight_for_quant, bit_assignment
                )
            else:
                # 统一精度量化
                weight_quantized = self.weight_quantizer(weight_for_quant, self.num_bits)
            
            weight_quantized = weight_quantized.transpose(1, 2).squeeze(0)
            
        elif self.granularity == 'per_group':
            # 分组量化
            weight_for_quant = weight.unsqueeze(0)  # [1, out_features, in_features]
            
            if self.use_adaptive and bit_assignment is not None:
                weight_quantized = self._mixed_precision_quantize_with_outliers(
                    weight_for_quant, bit_assignment
                )
            else:
                weight_quantized = self.weight_quantizer(weight_for_quant, self.num_bits)
            
            weight_quantized = weight_quantized.squeeze(0)
            
        else:  # per_tensor
            # 全局量化
            weight_flat = weight.view(1, 1, -1)
            
            if self.use_adaptive and bit_assignment is not None:
                weight_quantized = self._mixed_precision_quantize_with_outliers(
                    weight_flat, bit_assignment
                )
            else:
                weight_quantized = self.weight_quantizer(weight_flat, self.num_bits)
            
            weight_quantized = weight_quantized.view_as(weight)
        
        self.weight_quantized = weight_quantized
        
        # 统计离群值
        if self.use_outlier_clipping:
            self._update_outlier_stats(weight)
    
    def _mixed_precision_quantize_with_outliers(
        self, 
        weight: torch.Tensor,
        bit_assignment: torch.Tensor
    ) -> torch.Tensor:
        """
        混合精度量化，带离群值处理
        """
        out_features, in_features = weight.shape[1], weight.shape[2] if weight.dim() > 2 else weight.shape[1]
        weight_quantized = torch.zeros_like(weight)
        
        for g in range(self.num_groups):
            start_idx = g * (out_features // self.num_groups)
            end_idx = (g + 1) * (out_features // self.num_groups) if g < self.num_groups - 1 else out_features
            
            # 获取该组的权重
            if weight.dim() == 3:
                group_weight = weight[:, start_idx:end_idx, :]
            else:
                group_weight = weight[start_idx:end_idx]
            
            # 获取该组的比特数
            group_bits = int(bit_assignment[g].item())
            
            # 使用量化器量化该组
            group_quantized = self.weight_quantizer(group_weight, group_bits)
            
            if weight.dim() == 3:
                weight_quantized[:, start_idx:end_idx, :] = group_quantized
            else:
                weight_quantized[start_idx:end_idx] = group_quantized
        
        return weight_quantized
    
    def _update_outlier_stats(self, weight: torch.Tensor):
        """更新离群值统计"""
        clip_min, clip_max = self.weight_quantizer.compute_clip_bounds(weight)
        if clip_min is not None and clip_max is not None:
            outliers = ((weight < clip_min) | (weight > clip_max)).sum()
            self.num_outliers = outliers
            self.total_elements = weight.numel()
    
    def get_outlier_ratio(self) -> float:
        """获取离群值比例"""
        if self.total_elements > 0:
            return (self.num_outliers.float() / self.total_elements).item()
        return 0.0
    
    def _uniform_quantize(self, weight: torch.Tensor, num_bits: int) -> torch.Tensor:
        """
        统一精度量化
        """
        n_levels = 2 ** num_bits
        
        if self.symmetric:
            # 对称量化
            if self.per_channel:
                # Per-channel量化
                weight_abs_max = weight.abs().max(dim=1, keepdim=True)[0]
                weight_abs_max = weight_abs_max.clamp(min=1e-8)
                self.weight_scale = weight_abs_max / (n_levels // 2 - 1)
                self.weight_zero_point.zero_()
            else:
                # Per-tensor量化
                weight_abs_max = weight.abs().max()
                weight_abs_max = weight_abs_max.clamp(min=1e-8)
                self.weight_scale = weight_abs_max / (n_levels // 2 - 1)
                self.weight_zero_point.zero_()
            
            # 量化和反量化
            weight_normalized = weight / self.weight_scale
            weight_quantized = torch.clamp(
                torch.round(weight_normalized),
                -(n_levels // 2 - 1),
                n_levels // 2 - 1
            )
            weight_dequantized = weight_quantized * self.weight_scale
            
        else:
            # 非对称量化
            if self.per_channel:
                weight_min = weight.min(dim=1, keepdim=True)[0]
                weight_max = weight.max(dim=1, keepdim=True)[0]
            else:
                weight_min = weight.min()
                weight_max = weight.max()
            
            weight_range = weight_max - weight_min
            weight_range = weight_range.clamp(min=1e-8)
            
            self.weight_scale = weight_range / (n_levels - 1)
            self.weight_zero_point = -weight_min / self.weight_scale
            
            # 量化和反量化
            weight_normalized = weight / self.weight_scale + self.weight_zero_point
            weight_quantized = torch.clamp(torch.round(weight_normalized), 0, n_levels - 1)
            weight_dequantized = (weight_quantized - self.weight_zero_point) * self.weight_scale
        
        return weight_dequantized
    
    def _mixed_precision_quantize(
        self, 
        weight: torch.Tensor,
        bit_assignment: torch.Tensor
    ) -> torch.Tensor:
        """
        混合精度量化 - 不同组使用不同比特数
        
        Args:
            weight: [out_features, in_features]
            bit_assignment: [num_groups] 每组的比特数
        """
        out_features, in_features = weight.shape
        
        # 确定组的划分方式（这里按输出通道分组）
        group_size = out_features // self.num_groups
        weight_quantized = torch.zeros_like(weight)
        
        for g in range(self.num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size if g < self.num_groups - 1 else out_features
            
            # 获取该组的权重
            group_weight = weight[start_idx:end_idx]
            
            # 获取该组的比特数
            group_bits = int(bit_assignment[g].item())
            
            # 量化该组
            group_quantized = self._uniform_quantize(group_weight, group_bits)
            weight_quantized[start_idx:end_idx] = group_quantized
        
        return weight_quantized
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        """
        # 如果还没有量化，先量化
        if self.weight_quantized is None:
            self.quantize_weight()
        
        # 使用量化的权重进行计算
        output = F.linear(input, self.weight_quantized, self.bias)
        
        return output
    
    def extra_repr(self) -> str:
        """层的字符串表示"""
        return (f'in_features={self.in_features}, out_features={self.out_features}, '
                f'num_bits={self.num_bits}, per_channel={self.per_channel}, '
                f'num_groups={self.num_groups}')


class WeightQuantizer(nn.Module):
    """
    权重量化器 - 支持整个模型的权重量化
    """
    def __init__(
        self,
        model: nn.Module,
        target_bits: float = 4.0,
        num_groups: int = 8,
        quantize_embeddings: bool = False,
        quantize_lm_head: bool = False,
        skip_modules: Optional[List[str]] = None
    ):
        super().__init__()
        
        self.model = model
        self.target_bits = target_bits
        self.num_groups = num_groups
        self.quantize_embeddings = quantize_embeddings
        self.quantize_lm_head = quantize_lm_head
        self.skip_modules = skip_modules or []
        
        # 记录被量化的层
        self.quantized_layers = {}
        
        # 执行量化
        self._quantize_model()
    
    def _quantize_model(self):
        """
        遍历模型并量化Linear层
        """
        for name, module in self.model.named_modules():
            # 跳过指定的模块
            if any(skip in name for skip in self.skip_modules):
                continue
            
            # 跳过embedding和lm_head（如果指定）
            if not self.quantize_embeddings and 'embed' in name.lower():
                continue
            if not self.quantize_lm_head and 'lm_head' in name.lower():
                continue
            
            # 量化Linear层
            if isinstance(module, nn.Linear):
                # 获取父模块和属性名
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                parent = self.model
                
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                # 创建量化层
                quantized_layer = QuantizedLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    bias=module.bias is not None,
                    num_bits=int(self.target_bits),
                    per_channel=True,
                    symmetric=True,
                    num_groups=self.num_groups,
                    use_adaptive=True
                )
                
                # 复制权重
                quantized_layer.weight_fp.data = module.weight.data.clone()
                if module.bias is not None:
                    quantized_layer.bias.data = module.bias.data.clone()
                
                # 立即量化
                quantized_layer.quantize_weight()
                
                # 替换原始层
                setattr(parent, attr_name, quantized_layer)
                
                # 记录
                self.quantized_layers[name] = quantized_layer
                
                print(f"Quantized layer: {name} -> {self.target_bits} bits")
    
    def update_bit_allocation(self, bit_allocations: Dict[str, torch.Tensor]):
        """
        更新各层的比特分配
        
        Args:
            bit_allocations: {layer_name: bit_assignment_tensor}
        """
        for name, bit_assignment in bit_allocations.items():
            if name in self.quantized_layers:
                layer = self.quantized_layers[name]
                layer.quantize_weight(bit_assignment)
    
    def get_model_size(self) -> Dict[str, float]:
        """
        计算模型大小（MB）
        """
        original_size = 0
        quantized_size = 0
        
        for name, layer in self.quantized_layers.items():
            # 原始大小（FP32）
            param_count = layer.in_features * layer.out_features
            if layer.bias is not None:
                param_count += layer.out_features
            original_size += param_count * 4  # 4 bytes per FP32
            
            # 量化后大小
            if layer.use_adaptive:
                avg_bits = layer.bit_assignment.mean().item()
            else:
                avg_bits = layer.num_bits
            
            quantized_size += (layer.in_features * layer.out_features * avg_bits) / 8
            if layer.bias is not None:
                quantized_size += layer.out_features * 4  # bias通常不量化
        
        return {
            'original_mb': original_size / (1024 * 1024),
            'quantized_mb': quantized_size / (1024 * 1024),
            'compression_ratio': original_size / quantized_size if quantized_size > 0 else 0
        }


class IBWeightQuantizer(nn.Module):
    """
    基于信息瓶颈的权重量化器
    结合重要性估计自动分配比特
    """
    def __init__(
        self,
        model: nn.Module,
        importance_estimator: nn.Module,
        bit_allocator: nn.Module,
        target_bits: float = 4.0,
        num_groups: int = 8,
        beta: float = 0.01
    ):
        super().__init__()
        
        self.weight_quantizer = WeightQuantizer(
            model=model,
            target_bits=target_bits,
            num_groups=num_groups,
            quantize_embeddings=False,
            quantize_lm_head=False
        )
        
        self.importance_estimator = importance_estimator
        self.bit_allocator = bit_allocator
        self.beta = beta
    
    def compute_importance_for_weights(self, layer_name: str) -> torch.Tensor:
        """
        计算特定层权重的重要性分数
        """
        if layer_name not in self.weight_quantizer.quantized_layers:
            return None
        
        layer = self.weight_quantizer.quantized_layers[layer_name]
        weight = layer.weight_fp
        
        # 基于权重的统计量计算重要性
        # 1. 权重幅度
        weight_magnitude = weight.abs()
        
        # 2. 权重方差（每个输出通道）
        weight_variance = weight.var(dim=1, keepdim=True)
        
        # 3. 权重的信息熵
        weight_normalized = F.softmax(weight_magnitude, dim=1)
        weight_entropy = -(weight_normalized * torch.log(weight_normalized + 1e-8)).sum(dim=1, keepdim=True)
        
        # 组合重要性分数
        importance = weight_magnitude.mean(dim=1) * weight_variance.squeeze() * weight_entropy.squeeze()
        
        # 归一化到[0, 1]
        importance = (importance - importance.min()) / (importance.max() - importance.min() + 1e-8)
        
        return importance
    
    def allocate_bits_for_layer(self, layer_name: str, importance: torch.Tensor) -> torch.Tensor:
        """
        为层分配比特
        """
        # 使用bit_allocator
        allocation_result = self.bit_allocator(importance.unsqueeze(0))
        return allocation_result['bit_assignment'].squeeze(0)
    
    def quantize_all_weights(self):
        """
        量化所有权重层
        """
        total_compression = 0
        bit_allocations = {}
        
        for layer_name, layer in self.weight_quantizer.quantized_layers.items():
            # 计算重要性
            importance = self.compute_importance_for_weights(layer_name)
            
            if importance is not None and layer.use_adaptive:
                # 分配比特
                bit_assignment = self.allocate_bits_for_layer(layer_name, importance)
                bit_allocations[layer_name] = bit_assignment
                
                # 应用量化
                layer.quantize_weight(bit_assignment)
                
                # 统计
                avg_bits = bit_assignment.mean().item()
                print(f"Layer {layer_name}: avg bits = {avg_bits:.2f}")
        
        return bit_allocations
    
    def forward(self, *args, **kwargs):
        """前向传播（使用量化的模型）"""
        return self.weight_quantizer.model(*args, **kwargs)
