#!/usr/bin/env python3
"""
优化版 CIBD 模型
改进点：
1. IB应用到纯视觉特征（在mm_projector之后）
2. 多层特征蒸馏（浅层、中层、深层）
3. 余弦相似度损失替代MSE
4. 支持hidden_size压缩
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
import logging

logger = logging.getLogger(__name__)


class ImprovedVisualIB(nn.Module):
    """
    改进的视觉信息瓶颈
    应用在纯视觉特征上（mm_projector输出之后）
    """
    def __init__(self, input_dim, compression_ratio=0.7):
        super().__init__()
        
        bottleneck_dim = int(input_dim * compression_ratio)
        
        # VAE编码器（输出mu和log_var）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, bottleneck_dim * 2)
        )
        
        # VAE解码器
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim)
        )
        
        # 可学习的beta
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, visual_features):
        """
        Args:
            visual_features: [batch, num_patches, hidden_dim]
        Returns:
            compressed_features, ib_loss
        """
        batch_size, num_patches, hidden_dim = visual_features.shape
        
        # Flatten for processing
        x = visual_features.view(-1, hidden_dim)
        
        # Encode
        stats = self.encoder(x)
        mu, log_var = stats.chunk(2, dim=-1)
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # Reparameterize
        z = self.reparameterize(mu, log_var)
        
        # Decode
        x_recon = self.decoder(z)
        
        # Reshape back
        compressed_features = x_recon.view(batch_size, num_patches, hidden_dim)
        
        # Compute IB loss
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        recon_loss = F.mse_loss(x_recon, x)
        
        beta = torch.exp(self.log_beta)
        ib_loss = beta * kl_loss + recon_loss
        
        return compressed_features, {
            'ib_loss': ib_loss,
            'kl_loss': kl_loss,
            'recon_loss': recon_loss,
            'beta': beta
        }


class MultiLayerDistillation(nn.Module):
    """
    多层特征蒸馏
    在浅层、中层、深层分别进行蒸馏
    """
    def __init__(self, student_hidden, teacher_hidden, num_student_layers, num_teacher_layers):
        super().__init__()
        
        self.num_student_layers = num_student_layers
        self.num_teacher_layers = num_teacher_layers
        
        # 选择要对齐的层（浅层、中层、深层）
        self.student_layer_indices = [
            num_student_layers // 4,      # 浅层
            num_student_layers // 2,      # 中层
            3 * num_student_layers // 4   # 深层
        ]
        
        self.teacher_layer_indices = [
            num_teacher_layers // 4,
            num_teacher_layers // 2,
            3 * num_teacher_layers // 4
        ]
        
        # 如果维度不同，创建投影层
        self.projectors = nn.ModuleList()
        if student_hidden != teacher_hidden:
            for _ in range(3):  # 3个层级
                self.projectors.append(nn.Linear(student_hidden, teacher_hidden))
        else:
            for _ in range(3):
                self.projectors.append(nn.Identity())
        
        logger.info(f"多层蒸馏配置:")
        logger.info(f"  学生层: {self.student_layer_indices}")
        logger.info(f"  教师层: {self.teacher_layer_indices}")
    
    def cosine_similarity_loss(self, student_feat, teacher_feat):
        """
        余弦相似度损失
        鼓励特征方向对齐，而不是值对齐
        """
        # Normalize
        student_norm = F.normalize(student_feat, dim=-1)
        teacher_norm = F.normalize(teacher_feat, dim=-1)
        
        # Cosine similarity: 1 - cos(theta)
        cos_sim = (student_norm * teacher_norm).sum(dim=-1).mean()
        loss = 1.0 - cos_sim
        
        return loss
    
    def forward(self, student_hidden_states, teacher_hidden_states, attention_mask=None):
        """
        Args:
            student_hidden_states: List of [batch, seq_len, student_hidden]
            teacher_hidden_states: List of [batch, seq_len, teacher_hidden]
            attention_mask: [batch, seq_len]
        Returns:
            total_loss, loss_dict
        """
        total_loss = 0
        loss_dict = {}
        
        for i, (s_idx, t_idx) in enumerate(zip(self.student_layer_indices, self.teacher_layer_indices)):
            # 获取对应层的hidden states
            student_hidden = student_hidden_states[s_idx]
            teacher_hidden = teacher_hidden_states[t_idx]
            
            # 对齐序列长度
            min_len = min(student_hidden.size(1), teacher_hidden.size(1))
            student_hidden = student_hidden[:, :min_len, :]
            teacher_hidden = teacher_hidden[:, :min_len, :]
            
            # 投影学生特征（如果需要）
            student_hidden = self.projectors[i](student_hidden)
            
            # 应用attention mask（如果提供）
            if attention_mask is not None:
                mask = attention_mask[:, :min_len].unsqueeze(-1)
                student_hidden = student_hidden * mask
                teacher_hidden = teacher_hidden * mask
            
            # 计算余弦相似度损失
            layer_loss = self.cosine_similarity_loss(student_hidden, teacher_hidden)
            
            total_loss += layer_loss
            loss_dict[f'layer_{i}_loss'] = layer_loss.item()
        
        # 平均
        total_loss = total_loss / len(self.student_layer_indices)
        
        return total_loss, loss_dict


class LlavaCIBDModelV2(LlavaLlamaForCausalLM):
    """
    优化版 CIBD 模型
    
    改进：
    1. IB应用到纯视觉特征（encode_images之后）
    2. 多层特征蒸馏（3个层级）
    3. 余弦相似度损失
    4. 更好的损失权重平衡
    """
    
    def __init__(self, config, teacher_model=None):
        super().__init__(config)
        
        self.teacher_model = teacher_model
        self.config = config
        
        if teacher_model is not None:
            # 冻结教师模型
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            
            logger.info("✓ 教师模型已冻结")
        
        # ===== 1. 视觉信息瓶颈（应用在纯视觉特征上）=====
        self.visual_ib = ImprovedVisualIB(
            input_dim=config.hidden_size,
            compression_ratio=0.7  # 保留70%的信息
        )
        logger.info(f"✓ 视觉IB模块: {config.hidden_size} -> {int(config.hidden_size * 0.7)}")
        
        # ===== 2. 多层特征蒸馏 =====
        if teacher_model is not None:
            self.multi_layer_distill = MultiLayerDistillation(
                student_hidden=config.hidden_size,
                teacher_hidden=teacher_model.config.hidden_size,
                num_student_layers=config.num_hidden_layers,
                num_teacher_layers=teacher_model.config.num_hidden_layers
            )
            logger.info("✓ 多层蒸馏模块已创建")
        else:
            self.multi_layer_distill = None
        
        # ===== 3. Logits投影层（如果维度不同）=====
        if teacher_model and config.hidden_size != teacher_model.config.hidden_size:
            self.logits_projector = nn.Linear(config.hidden_size, teacher_model.config.hidden_size)
            logger.info(f"✓ Logits投影: {config.hidden_size} -> {teacher_model.config.hidden_size}")
        else:
            self.logits_projector = None
        
        # ===== 4. 损失权重（可学习）=====
        self.log_kd_weight = nn.Parameter(torch.tensor(0.0))  # exp(0) = 1.0
        self.log_feat_weight = nn.Parameter(torch.tensor(-0.69))  # exp(-0.69) ≈ 0.5
        self.log_ib_weight = nn.Parameter(torch.tensor(-2.3))  # exp(-2.3) ≈ 0.1
    
    def encode_images_with_ib(self, images):
        """
        带信息瓶颈的图像编码
        IB应用在纯视觉特征上
        """
        # 1. 标准视觉编码（vision_tower + mm_projector）
        image_features = self.encode_images(images)
        
        # 2. 维度适配（如果需要）
        if image_features.shape[-1] != self.config.hidden_size:
            # 创建适配层（只在第一次调用时创建）
            if not hasattr(self, 'visual_adapter'):
                self.visual_adapter = nn.Linear(
                    image_features.shape[-1], 
                    self.config.hidden_size
                ).to(image_features.device).to(image_features.dtype)
                logger.info(f"创建视觉适配层: {image_features.shape[-1]} -> {self.config.hidden_size}")
            
            image_features = self.visual_adapter(image_features)
        
        # 3. 应用信息瓶颈压缩
        compressed_features, ib_info = self.visual_ib(image_features)
        
        return compressed_features, ib_info
    
    def encode_images(self, images):
        """
        覆盖父类的图像编码方法
        
        【关键修复】：智能识别输入类型
        - 如果已经是压缩后的特征 [B, L, D]，直接返回（短路）
        - 如果是原始图片 [B, 3, H, W]，调用父类正常编码
        
        这样可以避免特征被二次送入视觉塔导致的通道错误
        """
        if images is None:
            return None
        
        # 检查是否已经是特征
        if torch.is_tensor(images) and images.dim() == 3:
            # [B, L, D] 格式，检查最后一维是否匹配 hidden_size
            if images.size(-1) == self.config.hidden_size:
                # 已经是压缩后的特征，直接返回
                return images
        
        # 否则是原始图片，调用父类的正常编码流程
        return super().encode_images(images)
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        images=None,
        image_sizes=None,
        return_dict=None,
    ):
        """
        优化的前向传播
        """
        
        # ===== 推理模式：标准前向 =====
        if not self.training or self.teacher_model is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=True,
                images=images,
                image_sizes=image_sizes,
                return_dict=True
            )
        
        # ===== 训练模式：完整蒸馏 =====
        
        # 保存原始输入（用于教师模型）
        original_input_ids = input_ids.clone() if input_ids is not None else None
        original_attention_mask = attention_mask.clone() if attention_mask is not None else None
        original_position_ids = position_ids.clone() if position_ids is not None else None
        original_labels = labels.clone() if labels is not None else None
        
        # ===== 学生路径：使用压缩的视觉特征 =====
        ib_info = None
        if images is not None:
            # 应用IB压缩
            compressed_images, ib_info = self.encode_images_with_ib(images)
            
            # 准备多模态输入（使用压缩后的视觉特征）
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                compressed_images,  # 使用压缩后的特征
                image_sizes
            )
        elif inputs_embeds is None:
            # 纯文本，准备输入
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                None,
                None
            )
        
        # 学生模型前向
        student_outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,  # 需要hidden states
            return_dict=True
        )
        
        # 初始化损失
        task_loss = student_outputs.loss if student_outputs.loss is not None else 0
        total_loss = task_loss
        
        # 损失记录
        loss_info = {
            'task_loss': task_loss.item() if torch.is_tensor(task_loss) else task_loss,
            'kl_loss': 0,
            'feature_loss': 0,
            'ib_loss': 0,
            'layer_0_loss': 0,
            'layer_1_loss': 0,
            'layer_2_loss': 0,
        }
        
        # ===== 1. 信息瓶颈损失 =====
        if ib_info is not None:
            ib_weight = torch.exp(self.log_ib_weight)
            ib_loss = ib_info['ib_loss']
            total_loss = total_loss + ib_weight * ib_loss
            
            loss_info['ib_loss'] = ib_loss.item()
            loss_info['ib_kl'] = ib_info['kl_loss'].item()
            loss_info['ib_recon'] = ib_info['recon_loss'].item()
            loss_info['ib_beta'] = ib_info['beta'].item()
            loss_info['ib_weight'] = ib_weight.item()
        
        # ===== 2. 知识蒸馏 =====
        if original_labels is not None:
            with torch.no_grad():
                # 教师模型前向（使用原始输入）
                teacher_outputs = self.teacher_model(
                    input_ids=original_input_ids,
                    attention_mask=original_attention_mask,
                    position_ids=original_position_ids,
                    images=images,  # 教师使用原始图像
                    image_sizes=image_sizes,
                    labels=original_labels,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # 2.1 Logits蒸馏（KL散度）
            temperature = 4.0
            student_logits = student_outputs.logits
            teacher_logits = teacher_outputs.logits
            
            # 对齐序列长度
            min_len = min(student_logits.size(1), teacher_logits.size(1))
            student_logits = student_logits[:, :min_len, :]
            teacher_logits = teacher_logits[:, :min_len, :]
            
            # 如果维度不同，投影学生logits
            if self.logits_projector is not None:
                # 需要先投影到最后一层hidden，再到vocab
                # 这里简化：直接在logits空间对齐（假设vocab相同）
                pass
            
            # KL散度
            student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
            teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
            
            kl_loss = F.kl_div(
                student_log_probs,
                teacher_probs,
                reduction='batchmean'
            ) * (temperature ** 2)
            
            kd_weight = torch.exp(self.log_kd_weight)
            total_loss = total_loss + kd_weight * kl_loss
            
            loss_info['kl_loss'] = kl_loss.item()
            loss_info['kd_weight'] = kd_weight.item()
            
            # 2.2 多层特征蒸馏（余弦相似度）
            if self.multi_layer_distill is not None:
                feature_loss, layer_losses = self.multi_layer_distill(
                    student_outputs.hidden_states,
                    teacher_outputs.hidden_states,
                    attention_mask=attention_mask
                )
                
                feat_weight = torch.exp(self.log_feat_weight)
                total_loss = total_loss + feat_weight * feature_loss
                
                loss_info['feature_loss'] = feature_loss.item()
                loss_info['feat_weight'] = feat_weight.item()
                loss_info.update(layer_losses)
        
        # ===== 返回 =====
        output = CausalLMOutputWithPast(
            loss=total_loss,
            logits=student_outputs.logits,
            past_key_values=student_outputs.past_key_values,
            hidden_states=student_outputs.hidden_states,
            attentions=student_outputs.attentions,
        )
        
        output.loss_info = loss_info
        
        return output