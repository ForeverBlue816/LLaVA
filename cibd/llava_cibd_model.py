import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Union
import copy

from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM, LlavaConfig
from transformers.modeling_outputs import CausalLMOutputWithPast


class LlavaCIBD(LlavaLlamaForCausalLM):
    """
    真正的CIBD压缩模型：
    1. 学生模型参数量真正减少
    2. 信息瓶颈用于指导压缩
    3. 保留原始文本序列
    """
    
    def __init__(self, config, teacher_model=None):
        # 调用父类构造函数，创建较小的学生模型
        super().__init__(config)
        
        self.teacher_model = teacher_model
        if teacher_model:
            # 冻结教师模型
            for param in self.teacher_model.parameters():
                param.requires_grad = False
            self.teacher_model.eval()
        
        # 添加信息瓶颈模块（用于视觉特征压缩）
        self.visual_compressor = VisualInformationBottleneck(
            input_dim=config.hidden_size,
            bottleneck_dim=config.hidden_size // 2,  # 压缩到一半
            output_dim=config.hidden_size
        )
        
        # 特征对齐投影（用于匹配教师模型维度）
        teacher_hidden_size = teacher_model.config.hidden_size if teacher_model else config.hidden_size
        if teacher_hidden_size != config.hidden_size:
            self.feature_projector = nn.Linear(config.hidden_size, teacher_hidden_size)
        else:
            self.feature_projector = nn.Identity()
            
        # 率失真权衡参数
        self.beta = nn.Parameter(torch.tensor(1.0))
        
    def encode_images_with_ib(self, images):
        """使用信息瓶颈编码图像"""
        # 原始视觉编码
        image_features = self.encode_images(images)
        
        # 通过信息瓶颈压缩
        compressed_features, ib_loss = self.visual_compressor(image_features)
        
        return compressed_features, ib_loss
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
    ):
        # 保存原始输入用于教师模型
        original_input_ids = input_ids
        original_images = images
        
        ib_loss = 0
        
        # 如果有图像，使用信息瓶颈处理
        if images is not None:
            # 学生路径：压缩图像特征
            compressed_image_features, ib_loss = self.encode_images_with_ib(images)
            
            # 准备多模态输入（保留文本序列）
            if inputs_embeds is None:
                (
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    inputs_embeds,
                    labels
                ) = self.prepare_inputs_labels_for_multimodal_compressed(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    compressed_image_features,  # 使用压缩后的特征
                    image_sizes
                )
        
        # 学生模型前向传播
        outputs = super(LlavaLlamaForCausalLM, self).forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True
        )
        
        total_loss = outputs.loss if outputs.loss is not None else 0
        
        # 添加信息瓶颈损失
        if ib_loss > 0:
            total_loss = total_loss + self.beta * ib_loss
        
        # 知识蒸馏
        if self.teacher_model is not None and labels is not None:
            with torch.no_grad():
                # 教师路径：使用原始输入（不压缩）
                teacher_outputs = self.teacher_model(
                    input_ids=original_input_ids,
                    attention_mask=attention_mask,
                    images=original_images,
                    labels=labels,
                    output_hidden_states=True,
                    return_dict=True
                )
            
            # KL散度蒸馏
            temperature = 4.0
            student_logits = outputs.logits / temperature
            teacher_logits = teacher_outputs.logits / temperature
            
            kl_loss = F.kl_div(
                F.log_softmax(student_logits, dim=-1),
                F.softmax(teacher_logits, dim=-1),
                reduction='batchmean'
            ) * (temperature ** 2)
            
            # 特征蒸馏（最后一层隐藏状态）
            student_hidden = outputs.hidden_states[-1]
            teacher_hidden = teacher_outputs.hidden_states[-1]
            
            # 对齐维度
            if student_hidden.size(-1) != teacher_hidden.size(-1):
                student_hidden = self.feature_projector(student_hidden)
            
            feature_loss = F.mse_loss(student_hidden, teacher_hidden)
            
            # 组合蒸馏损失
            distill_loss = 0.5 * kl_loss + 0.5 * feature_loss
            total_loss = total_loss + distill_loss
        
        if return_dict:
            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=outputs.logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )
        return (total_loss, outputs.logits)
    
    def prepare_inputs_labels_for_multimodal_compressed(
        self, input_ids, position_ids, attention_mask, 
        past_key_values, labels, compressed_image_features, image_sizes
    ):
        """
        准备压缩后的多模态输入，保留原始文本序列
        """
        # 这里重用父类的方法，但传入压缩后的图像特征
        # 关键：不覆盖文本tokens，只替换<image>位置的特征
        return self.prepare_inputs_labels_for_multimodal(
            input_ids, position_ids, attention_mask,
            past_key_values, labels, 
            compressed_image_features,  # 已压缩的特征
            image_sizes
        )


class VisualInformationBottleneck(nn.Module):
    """
    视觉信息瓶颈模块
    用于压缩视觉特征并计算率失真损失
    """
    def __init__(self, input_dim, bottleneck_dim, output_dim):
        super().__init__()
        
        # 编码器（压缩）
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, bottleneck_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # 解码器（重构）
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, output_dim),
            nn.ReLU()
        )
        
        self.bottleneck_dim = bottleneck_dim
        
    def reparameterize(self, mu, log_var):
        """VAE重参数化"""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        # 编码
        stats = self.encoder(x)
        mu, log_var = stats.chunk(2, dim=-1)
        
        # 采样
        z = self.reparameterize(mu, log_var)
        
        # 解码
        x_recon = self.decoder(z)
        
        # 计算IB损失
        # KL散度（率）
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=-1).mean()
        
        # 重构损失（失真）
        recon_loss = F.mse_loss(x_recon, x)
        
        # 总IB损失
        ib_loss = kl_loss + recon_loss
        
        return x_recon, ib_loss