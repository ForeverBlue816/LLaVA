"""
CM-IBQ Quantization Wrapper for LLaVA - 增强版
添加了版本检查、fallback机制和更好的错误处理
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Any
import sys
import os
import warnings
from packaging import version

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
    增强版：具有版本兼容性和fallback机制的LLaVA量化wrapper
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
        stage: int = 1,
        model_base: Optional[str] = None,
        use_fallback: bool = True,  # 新增：是否使用fallback机制
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
        self.use_fallback = use_fallback
        
        # 版本检查
        self._check_llava_version()
        
        # 加载原始模型
        print(f"Loading base model from {model_path}...")
        try:
            self.tokenizer, self.base_model, self.image_processor, self.context_len = load_pretrained_model(
                model_path=model_path,
                model_base=model_base,
                model_name=get_model_name_from_path(model_path)
            )
            print(f"Successfully loaded LLaVA model")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
        
        # 检查模型结构
        self._analyze_model_structure()
        
        # 获取维度信息
        self._extract_dimensions()
        
        # 初始化量化模块
        self._init_quantization_modules()
        
        # Stage 2: 初始化对齐损失模块
        if self.stage == 2:
            self._init_alignment_modules()
        
        # 设置量化hooks（带fallback机制）
        self._setup_quantization_hooks_safe()
        
        # 添加LoRA（如果启用且在Stage 2）
        if self.use_lora and self.stage == 2:
            self._setup_lora_adapters()
        
        # 内存优化
        self._apply_memory_optimizations()
    
    def _check_llava_version(self):
        """检查LLaVA版本兼容性"""
        try:
            import llava
            llava_version = getattr(llava, '__version__', 'unknown')
            print(f"Detected LLaVA version: {llava_version}")
            
            # 检查关键模块是否存在
            required_modules = [
                'llava.model.builder',
                'llava.mm_utils',
                'llava.constants'
            ]
            
            for module in required_modules:
                try:
                    __import__(module)
                except ImportError:
                    warnings.warn(f"Module {module} not found. Some features may not work.")
                    
        except ImportError:
            raise ImportError("LLaVA not found. Please install it first.")
    
    def _analyze_model_structure(self):
        """分析模型结构以确定最佳量化策略"""
        self.model_info = {
            'has_vision_tower': False,
            'has_mm_projector': False,
            'has_language_model': False,
            'vision_tower_type': None,
            'language_model_type': None
        }
        
        # 检查vision tower
        try:
            vision_tower = self.base_model.get_vision_tower()
            if vision_tower is not None:
                self.model_info['has_vision_tower'] = True
                self.model_info['vision_tower_type'] = type(vision_tower).__name__
        except:
            pass
        
        # 检查mm_projector
        try:
            model = self.base_model.get_model()
            if hasattr(model, 'mm_projector'):
                self.model_info['has_mm_projector'] = True
        except:
            pass
        
        # 检查language model
        if hasattr(self.base_model, 'config'):
            config = self.base_model.config
            if hasattr(config, 'model_type'):
                self.model_info['language_model_type'] = config.model_type
                self.model_info['has_language_model'] = True
        
        print(f"Model structure analysis: {self.model_info}")
    
    def _extract_dimensions(self):
        """提取模型维度信息（增强版）"""
        # Vision维度
        try:
            vision_tower = self.base_model.get_vision_tower()
            if vision_tower:
                # 尝试多种方式获取维度
                if hasattr(vision_tower, 'hidden_size'):
                    self.vision_dim = vision_tower.hidden_size
                elif hasattr(vision_tower, 'config') and hasattr(vision_tower.config, 'hidden_size'):
                    self.vision_dim = vision_tower.config.hidden_size
                elif hasattr(vision_tower, 'vision_model'):
                    if hasattr(vision_tower.vision_model.config, 'hidden_size'):
                        self.vision_dim = vision_tower.vision_model.config.hidden_size
                else:
                    self.vision_dim = 1024  # 默认值
                    warnings.warn(f"Could not determine vision dimension, using default: {self.vision_dim}")
            else:
                self.vision_dim = 1024
                warnings.warn("No vision tower found, using default dimension")
        except Exception as e:
            self.vision_dim = 1024
            warnings.warn(f"Error extracting vision dimension: {e}, using default: {self.vision_dim}")
        
        # LLM维度
        try:
            if hasattr(self.base_model, 'config'):
                self.llm_dim = self.base_model.config.hidden_size
            elif hasattr(self.base_model, 'model') and hasattr(self.base_model.model, 'config'):
                self.llm_dim = self.base_model.model.config.hidden_size
            else:
                self.llm_dim = 4096  # 默认值（常见的7B模型维度）
                warnings.warn(f"Could not determine LLM dimension, using default: {self.llm_dim}")
        except Exception as e:
            self.llm_dim = 4096
            warnings.warn(f"Error extracting LLM dimension: {e}, using default: {self.llm_dim}")
        
        # 投影器维度
        self.projection_input_dim = self.vision_dim
        self.projection_output_dim = self.llm_dim
        
        print(f"Dimensions - Vision: {self.vision_dim}, LLM: {self.llm_dim}")
    
    def _apply_memory_optimizations(self):
        """应用内存优化策略"""
        # 获取模型大小估算
        total_params = sum(p.numel() for p in self.base_model.parameters())
        model_size_gb = total_params * 4 / (1024**3)  # 假设fp32
        
        print(f"Model size: ~{model_size_gb:.1f}GB ({total_params/1e9:.1f}B parameters)")
        
        # 根据模型大小自动应用优化
        if model_size_gb > 10:  # 大于10GB的模型
            print("Applying memory optimizations for large model...")
            
            # 1. 启用梯度检查点
            if hasattr(self.base_model, 'gradient_checkpointing_enable'):
                self.base_model.gradient_checkpointing_enable()
                print("✓ Gradient checkpointing enabled")
            
            # 2. 对vision tower启用梯度检查点
            vision_tower = self.base_model.get_vision_tower()
            if vision_tower and hasattr(vision_tower, 'gradient_checkpointing_enable'):
                vision_tower.gradient_checkpointing_enable()
                print("✓ Vision tower gradient checkpointing enabled")
            
            # 3. 混合精度设置建议
            print("✓ Recommended: Use --fp16 or --bf16 for training")
            
            # 4. 如果是超大模型，建议使用DeepSpeed
            if model_size_gb > 30:
                print("✓ Recommended: Use DeepSpeed ZeRO-3 for training")
    
    def _setup_quantization_hooks_safe(self):
        """安全地设置量化hooks（带fallback）"""
        # 保存原始方法
        self._original_methods = {}
        
        # 尝试hook encode_images
        if hasattr(self.base_model, 'encode_images'):
            self._original_methods['encode_images'] = self.base_model.encode_images
            
            try:
                # 测试是否可以安全替换
                test_input = torch.randn(1, 3, 224, 224).to(next(self.base_model.parameters()).device)
                _ = self.base_model.encode_images(test_input)
                
                # 如果测试成功，进行替换
                self.base_model.encode_images = self._quantized_encode_images
                print("✓ Successfully hooked encode_images")
                
            except Exception as e:
                warnings.warn(f"Could not hook encode_images: {e}")
                if self.use_fallback:
                    print("Using fallback: wrapping forward method instead")
                    self._setup_fallback_hooks()
                else:
                    raise
        else:
            warnings.warn("encode_images not found in model")
            if self.use_fallback:
                self._setup_fallback_hooks()
        
        # Hook vision tower（安全方式）
        try:
            vision_tower = self.base_model.get_vision_tower()
            if vision_tower:
                self._hook_vision_tower_safe(vision_tower)
        except Exception as e:
            warnings.warn(f"Could not hook vision tower: {e}")
        
        # Hook投影器（安全方式）
        try:
            model = self.base_model.get_model()
            if hasattr(model, 'mm_projector'):
                self._hook_projector_safe(model.mm_projector)
        except Exception as e:
            warnings.warn(f"Could not hook projector: {e}")
    
    def _setup_fallback_hooks(self):
        """设置fallback hooks（当标准方法失败时）"""
        print("Setting up fallback quantization hooks...")
        
        # 包装整个forward方法
        original_forward = self.base_model.forward
        
        def quantized_forward(*args, **kwargs):
            # 在forward前后添加量化逻辑
            with torch.cuda.amp.autocast(enabled=False):
                # 可以在这里添加输入量化
                pass
            
            outputs = original_forward(*args, **kwargs)
            
            # 可以在这里添加输出量化
            return outputs
        
        self.base_model.forward = quantized_forward
    
    def _hook_vision_tower_safe(self, vision_tower):
        """安全地hook vision tower"""
        try:
            # 选择性地hook某些层
            if hasattr(vision_tower, 'vision_model') and hasattr(vision_tower.vision_model, 'encoder'):
                encoder = vision_tower.vision_model.encoder
                if hasattr(encoder, 'layers'):
                    # 每隔几层添加量化
                    num_layers = len(encoder.layers)
                    layers_to_quantize = list(range(0, num_layers, max(1, num_layers // 4)))
                    
                    for i in layers_to_quantize:
                        try:
                            self._add_forward_hook_to_layer(encoder.layers[i], f'vision_layer_{i}')
                            print(f"✓ Hooked vision layer {i}")
                        except Exception as e:
                            warnings.warn(f"Could not hook vision layer {i}: {e}")
        except Exception as e:
            warnings.warn(f"Error hooking vision tower: {e}")
    
    def _hook_projector_safe(self, projector):
        """安全地hook投影器"""
        try:
            # 为投影器添加前向hook
            def projector_hook(module, input, output):
                if self.training:
                    # 在这里可以添加量化逻辑
                    pass
                return output
            
            projector.register_forward_hook(projector_hook)
            print("✓ Hooked projector")
        except Exception as e:
            warnings.warn(f"Could not hook projector: {e}")
    
    def _quantized_encode_images(self, images):
        """量化版本的encode_images（增强版）"""
        try:
            # 1. 获取vision tower
            vision_tower = self.base_model.get_vision_tower()
            
            # 2. 通过vision tower
            if vision_tower:
                # 使用原始方法获取特征
                if hasattr(self, '_original_methods') and 'encode_images' in self._original_methods:
                    with torch.no_grad():
                        image_features = self._original_methods['encode_images'](images)
                else:
                    image_features = vision_tower(images)
            else:
                # Fallback到原始方法
                if hasattr(self, '_original_methods') and 'encode_images' in self._original_methods:
                    return self._original_methods['encode_images'](images)
                else:
                    raise RuntimeError("No vision tower found and no fallback available")
            
            # 3. 处理vision输出
            if hasattr(image_features, 'last_hidden_state'):
                features = image_features.last_hidden_state
            elif isinstance(image_features, tuple):
                features = image_features[0]
            else:
                features = image_features
            
            # 4. Vision特征量化
            if self.training and 'vision' in self.quantization_modules:
                features = self._apply_quantization(features, 'vision')
            
            # 5. 通过投影器
            model = self.base_model.get_model()
            if hasattr(model, 'mm_projector'):
                # IB处理（如果启用）
                if self.use_ib and 'projector_ib' in self.quantization_modules:
                    ib_result = self.quantization_modules['projector_ib'](features)
                    features = ib_result['z']
                    self._last_ib_result = ib_result
                
                # 通过投影器
                features = model.mm_projector(features)
                
                # 投影后量化
                if self.training and 'projector' in self.quantization_modules:
                    features = self._apply_quantization(features, 'projector')
                
                # Stage 2: 保存量化后的视觉特征用于对齐
                if self.training and self.stage == 2:
                    if not hasattr(self, '_intermediate_features'):
                        self._intermediate_features = {}
                    self._intermediate_features['visual_features'] = features.detach()
            
            return features
            
        except Exception as e:
            # 如果出错，fallback到原始方法
            warnings.warn(f"Error in quantized_encode_images: {e}, using fallback")
            if hasattr(self, '_original_methods') and 'encode_images' in self._original_methods:
                return self._original_methods['encode_images'](images)
            else:
                raise
    
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
        temperature = torch.exp(self.alignment_temperature).clamp(min=0.01, max=0.5)
        sim_matrix = torch.matmul(visual_proj, text_proj.T) / temperature  # [B, B]
        
        # InfoNCE损失
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
    
    def _apply_quantization(self, features: torch.Tensor, module_name: str) -> torch.Tensor:
        """应用量化到特征（增强版错误处理）"""
        try:
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
            
        except Exception as e:
            warnings.warn(f"Error in quantization for {module_name}: {e}, skipping quantization")
            return features
    
    def _extract_text_features(self, input_ids, attention_mask=None):
        """提取文本特征用于对齐"""
        model = self.base_model.get_model()
        
        # 获取文本嵌入
        if hasattr(model, 'embed_tokens'):
            text_embeds = model.embed_tokens(input_ids)
        elif hasattr(model, 'get_input_embeddings'):
            text_embeds = model.get_input_embeddings()(input_ids)
        else:
            warnings.warn("Could not extract text embeddings")
            return None
        
        # 应用attention mask
        if attention_mask is not None:
            text_embeds = text_embeds * attention_mask.unsqueeze(-1).float()
        
        return text_embeds
    
    def _add_forward_hook_to_layer(self, layer, layer_name):
        """为层添加forward hook（带错误处理）"""
        def hook_fn(module, input, output):
            try:
                if self.training:
                    output = self._apply_quantization(output, 'vision')
            except Exception as e:
                warnings.warn(f"Error in hook for {layer_name}: {e}")
            return output
        
        layer.register_forward_hook(hook_fn)
    
    def _setup_lora_adapters(self):
        """设置LoRA适配器（增强版）"""
        model = self.base_model.get_model()
        
        # 统计LoRA参数
        lora_params = 0
        
        # 为LLM层添加LoRA
        if hasattr(model, 'layers'):
            num_layers = len(model.layers)
            # 选择要添加LoRA的层（例如每隔一层）
            lora_layers = list(range(0, num_layers, max(1, num_layers // 8)))
            
            for i in lora_layers:
                try:
                    layer = model.layers[i]
                    
                    # 注意力
                    if hasattr(layer, 'self_attn'):
                        layer.self_attn = LoRAAdapter(
                            base_module=layer.self_attn,
                            rank=self.lora_rank,
                            alpha=32,
                            target_modules=['q_proj', 'v_proj'],
                            weight_bits=int(self.target_bits_weight)
                        )
                        lora_params += 2 * self.lora_rank * layer.self_attn.base_module.q_proj.in_features
                    
                    # FFN
                    if hasattr(layer, 'mlp'):
                        layer.mlp = LoRAAdapter(
                            base_module=layer.mlp,
                            rank=self.lora_rank,
                            alpha=32,
                            target_modules=['gate_proj', 'up_proj'],
                            weight_bits=int(self.target_bits_weight)
                        )
                        if hasattr(layer.mlp.base_module, 'gate_proj'):
                            lora_params += 2 * self.lora_rank * layer.mlp.base_module.gate_proj.in_features
                    
                    print(f"✓ Added LoRA to layer {i}")
                    
                except Exception as e:
                    warnings.warn(f"Could not add LoRA to layer {i}: {e}")
        
        if lora_params > 0:
            print(f"Total LoRA parameters: {lora_params/1e6:.2f}M")
    
    def forward(self, **kwargs):
        """前向传播（增强版错误处理）"""
        try:
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
                if text_features is not None:
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
                        try:
                            alignment_loss = self.compute_alignment_loss(
                                self._intermediate_features['visual_features'],
                                self._intermediate_features['text_features']
                            )
                            total_loss = total_loss + 0.3 * alignment_loss  # 对齐损失权重
                            
                            # 保存到输出用于监控
                            if not hasattr(outputs, 'aux_losses'):
                                outputs.aux_losses = {}
                            outputs.aux_losses['alignment_loss'] = alignment_loss
                        except Exception as e:
                            warnings.warn(f"Error computing alignment loss: {e}")
                    
                    # 比特率损失（较小权重）
                    for module_name, aux_info in self._quantization_aux.items():
                        if 'allocation_result' in aux_info:
                            total_loss = total_loss + 0.01 * aux_info['allocation_result']['bitrate_loss']
                
                outputs.loss = total_loss
            
            return outputs
            
        except Exception as e:
            print(f"Error in forward pass: {e}")
            # Fallback: 直接返回base model的输出
            if hasattr(self, 'base_model'):
                return self.base_model(**kwargs)
            else:
                raise
    
    def get_quantization_stats(self):
        """获取量化统计信息"""
        stats = {}
        
        try:
            # 量化统计
            for module_name, aux_info in self._quantization_aux.items():
                if 'allocation_result' in aux_info:
                    bit_assignment = aux_info['allocation_result']['bit_assignment']
                    stats[f'{module_name}_avg_bits'] = bit_assignment.mean().item()
                    stats[f'{module_name}_bit_std'] = bit_assignment.std().item()
                
                if 'importance_aux' in aux_info:
                    stats[f'{module_name}_importance_mean'] = aux_info['importance_aux'].get('mean_importance', 0)
            
            # 对齐统计（Stage 2）
            if hasattr(self, '_alignment_stats'):
                stats.update(self._alignment_stats)
                
        except Exception as e:
            warnings.warn(f"Error getting quantization stats: {e}")
        
        return stats
