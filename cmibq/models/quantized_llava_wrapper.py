"""
CM-IBQ Quantization Wrapper for LLaVA - 改进版
与core模块正确集成，包含对齐损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import sys
import os

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

# 正确导入core模块
from ..core.ib_quantizer import IBQuantizedLayer
from ..core.importance_estimator import HybridImportanceEstimation
from ..core.bit_allocator import BitAllocationNetwork  
from ..core.differentiable_quant import DifferentiableQuantizer
from .lora_adapter import LoRAAdapter


class CMIBQQuantizedLLaVA(nn.Module):
    """
    改进版：正确集成所有core模块的LLaVA量化wrapper
    """
    
    def __init__(
        self,
        model_path: str,
        target_bits_act: float = 4.0,
        target_bits_weight: float = 4.0,
        use_ib: bool = True,
        use_lora: bool = True,
        lora_rank: int = 16,
        num_groups: int = 8,
        stage: int = 1,  # 添加stage参数
        model_base: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        
        self.target_bits_act = target_bits_act
        self.target_bits_weight = target_bits_weight
        self.use_ib = use_ib
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.num_groups = num_groups
        self.stage = stage
        
        # 加载原始模型
        print(f"Loading base model from {model_path}...")
        self.tokenizer, self.base_model, self.image_processor, self.context_len = load_pretrained_model(
            model_path=model_path,
            model_base=model_base,
            model_name=get_model_name_from_path(model_path)
        )
        
        # 获取维度信息
        self._extract_dimensions()
        
        # 初始化量化模块
        self._init_quantization_modules()
        
        # Stage 2: 初始化对齐损失模块
        if self.stage == 2:
            self._init_alignment_modules()
        
        # 设置量化hooks
        self._setup_quantization_hooks()
        
        # 添加LoRA（如果启用且在Stage 2）
        if self.use_lora and self.stage == 2:
            self._setup_lora_adapters()
    
    def _extract_dimensions(self):
        """提取模型维度信息"""
        # Vision维度
        vision_tower = self.base_model.get_vision_tower()
        if vision_tower:
            self.vision_dim = vision_tower.hidden_size if hasattr(vision_tower, 'hidden_size') else 1024
        else:
            self.vision_dim = 1024  # 默认值
        
        # LLM维度
        self.llm_dim = self.base_model.config.hidden_size
        
        # 投影器维度
        model = self.base_model.get_model()
        if hasattr(model, 'mm_projector'):
            # 通常是 vision_dim -> llm_dim
            self.projection_input_dim = self.vision_dim
            self.projection_output_dim = self.llm_dim
        
        print(f"Dimensions - Vision: {self.vision_dim}, LLM: {self.llm_dim}")
    
    def _init_alignment_modules(self):
        """初始化对齐损失相关模块"""
        print("Initializing alignment modules for Stage 2...")
        
        # 对比学习温度参数
        self.alignment_temperature = nn.Parameter(torch.tensor(0.07))
        
        # 视觉和文本的投影头（用于对齐）
        self.visual_proj_head = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.ReLU(),
            nn.Linear(self.llm_dim, self.llm_dim // 2),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 2, 256)  # 投影到低维空间
        )
        
        self.text_proj_head = nn.Sequential(
            nn.Linear(self.llm_dim, self.llm_dim),
            nn.ReLU(),
            nn.Linear(self.llm_dim, self.llm_dim // 2),
            nn.ReLU(),
            nn.Linear(self.llm_dim // 2, 256)  # 投影到低维空间
        )
    
    def _init_quantization_modules(self):
        """初始化所有量化相关模块"""
        self.quantization_modules = nn.ModuleDict()
        
        # 1. Vision Tower量化
        self.quantization_modules['vision'] = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.vision_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.vision_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_act,
                bit_levels=[2, 4, 8]
            ),
            'quantizer': DifferentiableQuantizer(
                feature_dim=self.vision_dim,
                per_sample=True,
                bit_levels=[2, 4, 8],
                target_budget=self.target_bits_act
            )
        })
        
        # 2. 投影器量化（最关键）
        if self.use_ib:
            # 使用IB框架
            self.quantization_modules['projector_ib'] = IBQuantizedLayer(
                input_dim=self.projection_input_dim,
                bottleneck_dim=self.projection_output_dim,
                target_bits=self.target_bits_act,
                adaptive_beta=True
            )
        
        self.quantization_modules['projector'] = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.projection_output_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.projection_output_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_act,
                bit_levels=[2, 4, 8]
            ),
            'quantizer': DifferentiableQuantizer(
                feature_dim=self.projection_output_dim,
                per_sample=True,
                bit_levels=[2, 4, 8],
                target_budget=self.target_bits_act
            )
        })
        
        # 3. LLM层量化（选择性）
        self.quantization_modules['llm'] = nn.ModuleDict({
            'importance': HybridImportanceEstimation(
                feature_dim=self.llm_dim,
                num_groups=self.num_groups
            ),
            'bit_allocator': BitAllocationNetwork(
                feature_dim=self.llm_dim,
                num_groups=self.num_groups,
                target_bits=self.target_bits_act,
                bit_levels=[2, 4, 8]
            ),
            'quantizer': DifferentiableQuantizer(
                feature_dim=self.llm_dim,
                per_sample=True,
                bit_levels=[2, 4, 8],
                target_budget=self.target_bits_act
            )
        })
    
    def compute_alignment_loss(self, visual_features: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        """
        使用InfoNCE计算视觉-文本对齐损失
        
        Args:
            visual_features: [B, T_v, D] 量化后的视觉特征
            text_features: [B, T_t, D] 文本特征
        """
        batch_size = visual_features.size(0)
        
        # 池化到序列级别表示
        # 使用注意力池化或平均池化
        visual_pooled = visual_features.mean(dim=1)  # [B, D]
        
        # 对于文本，使用最后一个有效token（或平均）
        if text_features.dim() == 3:
            text_pooled = text_features.mean(dim=1)  # [B, D]
        else:
            text_pooled = text_features  # 已经是[B, D]
        
        # 通过投影头
        visual_proj = self.visual_proj_head(visual_pooled)  # [B, 256]
        text_proj = self.text_proj_head(text_pooled)  # [B, 256]
        
        # L2归一化
        visual_proj = F.normalize(visual_proj, p=2, dim=-1)
        text_proj = F.normalize(text_proj, p=2, dim=-1)
        
        # 计算相似度矩阵
        # 使用可学习的温度参数
        temperature = torch.exp(self.alignment_temperature).clamp(min=0.01, max=0.5)
        sim_matrix = torch.matmul(visual_proj, text_proj.T) / temperature  # [B, B]
        
        # InfoNCE损失
        # 对角线元素是正样本对，其他是负样本
        labels = torch.arange(batch_size, device=sim_matrix.device)
        
        # 视觉到文本的损失
        loss_v2t = F.cross_entropy(sim_matrix, labels)
        
        # 文本到视觉的损失
        loss_t2v = F.cross_entropy(sim_matrix.T, labels)
        
        # 总对齐损失
        alignment_loss = (loss_v2t + loss_t2v) / 2
        
        # 计算准确率用于监控
        with torch.no_grad():
            pred_v2t = sim_matrix.argmax(dim=1)
            pred_t2v = sim_matrix.T.argmax(dim=1)
            acc_v2t = (pred_v2t == labels).float().mean()
            acc_t2v = (pred_t2v == labels).float().mean()
            
            # 保存到辅助信息
            if not hasattr(self, '_alignment_stats'):
                self._alignment_stats = {}
            self._alignment_stats['acc_v2t'] = acc_v2t.item()
            self._alignment_stats['acc_t2v'] = acc_t2v.item()
            self._alignment_stats['temperature'] = temperature.item()
        
        return alignment_loss
    
    def _setup_quantization_hooks(self):
        """设置量化hooks"""
        # Hook到encode_images
        self._original_encode_images = self.base_model.encode_images
        self.base_model.encode_images = self._quantized_encode_images
        
        # Hook到vision tower的forward
        vision_tower = self.base_model.get_vision_tower()
        if vision_tower:
            self._hook_vision_tower(vision_tower)
        
        # Hook到投影器
        model = self.base_model.get_model()
        if hasattr(model, 'mm_projector'):
            self._hook_projector(model.mm_projector)
    
    def _quantized_encode_images(self, images):
        """量化版本的encode_images"""
        # 1. 获取vision tower
        vision_tower = self.base_model.get_vision_tower()
        
        # 2. 通过vision tower（会触发我们的hook）
        if vision_tower:
            image_features = vision_tower(images)
        else:
            image_features = self._original_encode_images(images)
            return image_features
        
        # 3. 处理vision输出
        if hasattr(image_features, 'last_hidden_state'):
            features = image_features.last_hidden_state
        else:
            features = image_features
        
        # 4. Vision特征量化
        features = self._apply_quantization(features, 'vision')
        
        # 5. 通过投影器
        model = self.base_model.get_model()
        if hasattr(model, 'mm_projector'):
            # IB处理（如果启用）
            if self.use_ib and 'projector_ib' in self.quantization_modules:
                ib_result = self.quantization_modules['projector_ib'](features)
                features = ib_result['z']
                self._last_ib_result = ib_result  # 保存用于损失计算
            
            # 通过投影器
            features = model.mm_projector(features)
            
            # 投影后量化
            features = self._apply_quantization(features, 'projector')
            
            # Stage 2: 保存量化后的视觉特征用于对齐
            if self.training and self.stage == 2:
                if not hasattr(self, '_intermediate_features'):
                    self._intermediate_features = {}
                self._intermediate_features['visual_features'] = features.detach()
        
        return features
    
    def _apply_quantization(self, features: torch.Tensor, module_name: str) -> torch.Tensor:
        """应用量化到特征"""
        if not self.training or module_name not in self.quantization_modules:
            return features
        
        modules = self.quantization_modules[module_name]
        
        # 处理维度
        original_shape = features.shape
        if features.dim() == 2:  # [B, D]
            features = features.unsqueeze(1)  # [B, 1, D]
        
        # 1. 计算重要性
        importance_scores, importance_aux = modules['importance'](features)
        
        # 2. 分配比特
        allocation_result = modules['bit_allocator'](importance_scores)
        
        # 3. 执行量化
        features_quantized = modules['quantizer'](
            features,
            bit_assignment=allocation_result['bit_assignment'],
            group_indices=allocation_result['group_indices']
        )
        
        # 恢复原始形状
        if len(original_shape) == 2:
            features_quantized = features_quantized.squeeze(1)
        
        # 保存辅助信息用于损失计算
        if not hasattr(self, '_quantization_aux'):
            self._quantization_aux = {}
        self._quantization_aux[module_name] = {
            'importance_aux': importance_aux,
            'allocation_result': allocation_result
        }
        
        return features_quantized
    
    def _extract_text_features(self, input_ids, attention_mask=None):
        """提取文本特征用于对齐"""
        model = self.base_model.get_model()
        
        # 获取文本嵌入
        if hasattr(model, 'embed_tokens'):
            text_embeds = model.embed_tokens(input_ids)
        else:
            text_embeds = model.get_input_embeddings()(input_ids)
        
        # 应用attention mask
        if attention_mask is not None:
            text_embeds = text_embeds * attention_mask.unsqueeze(-1).float()
        
        return text_embeds
    
    def _hook_vision_tower(self, vision_tower):
        """Hook vision tower的特定层"""
        # 选择性地hook某些层
        if hasattr(vision_tower, 'vision_model') and hasattr(vision_tower.vision_model, 'encoder'):
            encoder = vision_tower.vision_model.encoder
            if hasattr(encoder, 'layers'):
                # 每隔几层添加量化
                for i in range(0, len(encoder.layers), 3):
                    self._add_forward_hook_to_layer(encoder.layers[i], f'vision_layer_{i}')
    
    def _hook_projector(self, projector):
        """Hook投影器"""
        # 已经在_quantized_encode_images中处理
        pass
    
    def _add_forward_hook_to_layer(self, layer, layer_name):
        """为层添加forward hook"""
        def hook_fn(module, input, output):
            if self.training:
                output = self._apply_quantization(output, 'vision')
            return output
        
        layer.register_forward_hook(hook_fn)
    
    def _setup_lora_adapters(self):
        """设置LoRA适配器"""
        model = self.base_model.get_model()
        
        # 为LLM层添加LoRA
        if hasattr(model, 'layers'):
            for i, layer in enumerate(model.layers):
                # 选择性地添加LoRA
                if i % 2 == 0:  # 每隔一层
                    # 注意力
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn = LoRAAdapter(
                            base_module=layer.self_attn,
                            rank=self.lora_rank,
                            alpha=32,
                            target_modules=['q_proj', 'v_proj'],
                            weight_bits=int(self.target_bits_weight)
                        )
                    
                    # FFN
                    if hasattr(layer, 'mlp'):
                        layer.mlp = LoRAAdapter(
                            base_module=layer.mlp,
                            rank=self.lora_rank,
                            alpha=32,
                            target_modules=['gate_proj', 'up_proj'],
                            weight_bits=int(self.target_bits_weight)
                        )
    
    def forward(self, **kwargs):
        """前向传播"""
        # 清空辅助信息
        self._quantization_aux = {}
        self._intermediate_features = {}
        self._alignment_stats = {}
        
        # Stage 2: 提取文本特征用于对齐
        if self.training and self.stage == 2 and 'input_ids' in kwargs:
            text_features = self._extract_text_features(
                kwargs['input_ids'],
                kwargs.get('attention_mask')
            )
            self._intermediate_features['text_features'] = text_features.detach()
        
        # 调用base model
        outputs = self.base_model(**kwargs)
        
        # 添加各种损失
        if self.training and hasattr(outputs, 'loss'):
            total_loss = outputs.loss  # 保留原始任务损失
            
            # Stage 1: IB损失（主要）
            if self.stage == 1:
                if hasattr(self, '_last_ib_result'):
                    total_loss = total_loss + 0.5 * self._last_ib_result['total_loss']
                
                # 比特率损失
                for module_name, aux_info in self._quantization_aux.items():
                    if 'allocation_result' in aux_info:
                        total_loss = total_loss + 0.1 * aux_info['allocation_result']['bitrate_loss']
                        total_loss = total_loss + 0.01 * aux_info['allocation_result']['diversity_bonus']
            
            # Stage 2: 对齐损失（重要）
            elif self.stage == 2:
                # 对齐损失
                if 'visual_features' in self._intermediate_features and 'text_features' in self._intermediate_features:
                    alignment_loss = self.compute_alignment_loss(
                        self._intermediate_features['visual_features'],
                        self._intermediate_features['text_features']
                    )
                    total_loss = total_loss + 0.3 * alignment_loss  # 对齐损失权重
                    
                    # 保存到输出用于监控
                    if not hasattr(outputs, 'aux_losses'):
                        outputs.aux_losses = {}
                    outputs.aux_losses['alignment_loss'] = alignment_loss
                
                # 比特率损失（较小权重）
                for module_name, aux_info in self._quantization_aux.items():
                    if 'allocation_result' in aux_info:
                        total_loss = total_loss + 0.01 * aux_info['allocation_result']['bitrate_loss']
            
            outputs.loss = total_loss
        
        return outputs
    
    def get_quantization_stats(self):
        """获取量化统计信息"""
        stats = {}
        
        # 量化统计
        for module_name, aux_info in self._quantization_aux.items():
            if 'allocation_result' in aux_info:
                bit_assignment = aux_info['allocation_result']['bit_assignment']
                stats[f'{module_name}_avg_bits'] = bit_assignment.mean().item()
                stats[f'{module_name}_bit_std'] = bit_assignment.std().item()
            
            if 'importance_aux' in aux_info:
                stats[f'{module_name}_importance_mean'] = aux_info['importance_aux']['mean_importance']
        
        # 对齐统计（Stage 2）
        if hasattr(self, '_alignment_stats'):
            stats.update(self._alignment_stats)
        
        return stats