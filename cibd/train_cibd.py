#!/usr/bin/env python3
"""
CIBD训练脚本 - 完整版
包含所有蒸馏、压缩、日志和checkpoint功能
"""

import os
import sys
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup
import argparse
from tqdm import tqdm
import logging
from typing import Optional, Dict, Any

# 导入LLaVA组件
from llava.model.builder import load_pretrained_model
from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

# 导入CIBD组件
from cibd.dataset import LLaVADataset, LLaVACollator
from cibd.create_student_model import (
    create_compressed_student_config, 
    initialize_student_from_teacher,
    analyze_compression
)

# 设置日志
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


class LlavaCIBDModel(LlavaLlamaForCausalLM):
    """
    CIBD蒸馏模型 - 继承自LlavaLlamaForCausalLM
    完全兼容LLaVA的所有功能
    """
    
    def __init__(self, config, teacher_model=None):
        super().__init__(config)
        
        self.teacher_model = teacher_model
        self.is_student = True
        
        if teacher_model is not None:
            # 冻结教师模型
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
        
        # 信息瓶颈模块 - VAE风格
        hidden_size = config.hidden_size
        
        # 编码器：输出 mu 和 log_var
        self.visual_compressor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.LayerNorm(hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # 解码器：从隐变量重构
        self.visual_decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        # 可学习的率失真权重
        self.log_beta = nn.Parameter(torch.tensor(0.0))
        
        # 特征对齐投影（如果维度不同）
        if teacher_model and teacher_model.config.hidden_size != hidden_size:
            teacher_hidden = teacher_model.config.hidden_size
            self.feature_projector = nn.Linear(hidden_size, teacher_hidden)
            logger.info(f"添加特征投影: {hidden_size} -> {teacher_hidden}")
        else:
            self.feature_projector = None
    
    def compress_visual_features(self, visual_features):
        """
        使用信息瓶颈压缩视觉特征
        返回: (压缩后的特征, IB损失)
        """
        # VAE编码
        stats = self.visual_compressor(visual_features)
        mu, log_var = stats.chunk(2, dim=-1)
        
        # 限制log_var范围，避免数值不稳定
        log_var = torch.clamp(log_var, min=-10, max=10)
        
        # 重参数化
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        
        # 解码
        compressed = self.visual_decoder(z)
        
        # 计算IB损失
        # 1. KL散度（信息率）
        kl_loss = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())
        
        # 2. 重构损失（失真）
        recon_loss = nn.functional.mse_loss(compressed, visual_features)
        
        ib_loss = kl_loss + recon_loss
        
        return compressed, ib_loss
    
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
            重写forward方法，添加CIBD逻辑
            """
            
            # 推理模式或无教师模型：使用标准forward
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
            
            # === 训练模式：CIBD蒸馏 ===
            
            # ===== 关键修复：保存原始 input_ids =====
            original_input_ids = input_ids.clone() if input_ids is not None else None
            original_attention_mask = attention_mask.clone() if attention_mask is not None else None
            original_position_ids = position_ids.clone() if position_ids is not None else None
            original_labels = labels.clone() if labels is not None else None
            # ===== 修复结束 =====
            
            # 1. 处理多模态输入
            if images is not None and inputs_embeds is None:
                # 使用LLaVA的标准方法处理图像
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
                    images,
                    image_sizes
                )
            
            # 2. 学生模型前向传播
            student_outputs = super().forward(
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
            
            # 基础任务损失
            total_loss = student_outputs.loss if student_outputs.loss is not None else 0
            task_loss = total_loss.item() if torch.is_tensor(total_loss) else total_loss
            
            # 3. 信息瓶颈损失
            ib_loss = 0
            if images is not None and len(student_outputs.hidden_states) > 0:
                # 使用第一层hidden state
                first_hidden = student_outputs.hidden_states[0]
                
                # 随机采样减少计算
                sample_size = min(100, first_hidden.size(1))
                sample_indices = torch.randint(
                    0, first_hidden.size(1), 
                    (sample_size,),
                    device=first_hidden.device
                )
                sampled_hidden = first_hidden[:, sample_indices, :]
                
                _, ib_loss = self.compress_visual_features(sampled_hidden)
                
                # 添加到总损失
                beta = torch.exp(self.log_beta)
                total_loss = total_loss + 0.1 * beta * ib_loss
            
            # 4. 知识蒸馏
            kl_loss = 0
            feature_loss = 0
            
            if original_labels is not None:
                with torch.no_grad():
                    # ===== 关键修复：使用原始的 input_ids =====
                    # 教师模型前向传播
                    teacher_outputs = self.teacher_model(
                        input_ids=original_input_ids,  # ← 使用保存的原始 input_ids
                        attention_mask=original_attention_mask,  # ← 使用原始的
                        position_ids=original_position_ids,  # ← 使用原始的
                        images=images,
                        image_sizes=image_sizes,
                        labels=original_labels,  # ← 使用原始的 labels
                        output_hidden_states=True,
                        return_dict=True
                    )
                    # ===== 修复结束 =====
                
                # 4.1 Logits蒸馏（KL散度）
                temperature = 4.0
                student_logits = student_outputs.logits
                teacher_logits = teacher_outputs.logits
                
                # 对齐序列长度
                min_len = min(student_logits.size(1), teacher_logits.size(1))
                student_logits = student_logits[:, :min_len, :]
                teacher_logits = teacher_logits[:, :min_len, :]
                
                # 只在有效位置计算KL（排除padding）
                if attention_mask is not None:
                    mask = attention_mask[:, :min_len].unsqueeze(-1)
                    student_logits = student_logits * mask
                    teacher_logits = teacher_logits * mask
                
                # 计算KL散度
                student_log_probs = torch.nn.functional.log_softmax(student_logits / temperature, dim=-1)
                teacher_probs = torch.nn.functional.softmax(teacher_logits / temperature, dim=-1)
                
                kl_loss = torch.nn.functional.kl_div(
                    student_log_probs,
                    teacher_probs,
                    reduction='batchmean'
                ) * (temperature ** 2)
                
                # 添加到总损失
                total_loss = total_loss + 0.5 * kl_loss
                
                # 4.2 特征蒸馏
                if len(student_outputs.hidden_states) > 0 and len(teacher_outputs.hidden_states) > 0:
                    # 选择中间层进行对齐
                    student_hidden = student_outputs.hidden_states[len(student_outputs.hidden_states) // 2]
                    teacher_hidden = teacher_outputs.hidden_states[len(teacher_outputs.hidden_states) // 2]
                    
                    # 对齐序列长度
                    min_seq_len = min(student_hidden.size(1), teacher_hidden.size(1))
                    student_hidden = student_hidden[:, :min_seq_len, :]
                    teacher_hidden = teacher_hidden[:, :min_seq_len, :]
                    
                    # 如果维度不同，使用投影层
                    if self.feature_projector is not None:
                        student_hidden = self.feature_projector(student_hidden)
                    
                    # MSE损失
                    feature_loss = torch.nn.functional.mse_loss(student_hidden, teacher_hidden)
                    total_loss = total_loss + 0.3 * feature_loss
            
            # 返回结果
            return CausalLMOutputWithPast(
                loss=total_loss,
                logits=student_outputs.logits,
                past_key_values=student_outputs.past_key_values,
                hidden_states=student_outputs.hidden_states,
                attentions=student_outputs.attentions,
            )
            
            # 附加自定义损失信息（用于日志）
            output.loss_info = {
                    'total_loss': total_loss.item() if torch.is_tensor(total_loss) else total_loss,
                    'task_loss': task_loss,
                    'kl_loss': kl_loss.item() if torch.is_tensor(kl_loss) else kl_loss,
                    'feature_loss': feature_loss.item() if torch.is_tensor(feature_loss) else feature_loss,
                    'ib_loss': ib_loss.item() if torch.is_tensor(ib_loss) else ib_loss,
                    'beta': torch.exp(self.log_beta).item()
                }
                
            return output
        
            return (total_loss, student_outputs.logits)


class CIBDTrainer:
    """CIBD训练器"""
    
    def __init__(self, model, train_loader, val_loader, args, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        self.device = device
        
        self.model.to(device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            betas=(0.9, 0.999)
        )
        
        # 学习率调度器
        total_steps = len(train_loader) * args.num_epochs
        warmup_steps = int(total_steps * 0.03)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 混合精度
        self.scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        
        # 追踪
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # 日志累积
        self.loss_accumulator = {
            'total_loss': 0,
            'task_loss': 0,
            'kl_loss': 0,
            'feature_loss': 0,
            'ib_loss': 0,
            'beta': 0,
            'count': 0
        }
    
    def train_step(self, batch):
        """单步训练"""
        self.model.train()
        
        # 准备数据
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # 混合精度前向
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                labels=batch['labels'],
                images=batch.get('images'),
                image_sizes=batch.get('image_sizes'),
                return_dict=True
            )
            
            loss = outputs.loss / self.args.gradient_accumulation_steps
        
        # 反向传播
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        # 累积损失信息
        if hasattr(outputs, 'loss_info'):
            for key, value in outputs.loss_info.items():
                if key in self.loss_accumulator:
                    self.loss_accumulator[key] += value
            self.loss_accumulator['count'] += 1
        
        return loss.item() * self.args.gradient_accumulation_steps
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        
        # 重置累积器
        for key in self.loss_accumulator:
            self.loss_accumulator[key] = 0
        
        progress_bar = tqdm(
            self.train_loader, 
            desc=f"Epoch {epoch+1}/{self.args.num_epochs}"
        )
        
        for step, batch in enumerate(progress_bar):
            # 训练步骤
            loss = self.train_step(batch)
            
            # 梯度更新
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                # 优化器步进
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # 更新进度条
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': f"{loss:.4f}",
                    'lr': f"{current_lr:.2e}"
                })
            
            # 定期日志
            if self.global_step % 100 == 0 and self.loss_accumulator['count'] > 0:
                self.log_losses()
        
        # Epoch结束时计算平均损失
        avg_losses = self.get_average_losses()
        return avg_losses['total_loss']
    
    def validate(self):
        """验证"""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0
        
        # 重置累积器
        val_accumulator = {
            'total_loss': 0,
            'task_loss': 0,
            'kl_loss': 0,
            'feature_loss': 0,
            'ib_loss': 0,
            'count': 0
        }
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        labels=batch['labels'],
                        images=batch.get('images'),
                        image_sizes=batch.get('image_sizes'),
                        return_dict=True
                    )
                
                total_loss += outputs.loss.item()
                
                # 累积验证损失
                if hasattr(outputs, 'loss_info'):
                    for key, value in outputs.loss_info.items():
                        if key in val_accumulator:
                            val_accumulator[key] += value
                    val_accumulator['count'] += 1
        
        # 打印验证损失分解
        if val_accumulator['count'] > 0:
            logger.info("验证损失分解:")
            for key in ['task_loss', 'kl_loss', 'feature_loss', 'ib_loss']:
                avg = val_accumulator[key] / val_accumulator['count']
                logger.info(f"  {key}: {avg:.4f}")
        
        return total_loss / len(self.val_loader)
    
    def log_losses(self):
        """记录详细损失"""
        if self.loss_accumulator['count'] == 0:
            return
        
        logger.info(f"\n[Step {self.global_step}] 损失分解:")
        for key in ['total_loss', 'task_loss', 'kl_loss', 'feature_loss', 'ib_loss', 'beta']:
            avg = self.loss_accumulator[key] / self.loss_accumulator['count']
            logger.info(f"  {key}: {avg:.4f}")
        
        # 重置累积器
        for key in self.loss_accumulator:
            self.loss_accumulator[key] = 0
    
    def get_average_losses(self):
        """获取平均损失"""
        if self.loss_accumulator['count'] == 0:
            return {'total_loss': 0}
        
        return {
            key: self.loss_accumulator[key] / self.loss_accumulator['count']
            for key in self.loss_accumulator if key != 'count'
        }
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'args': vars(self.args)
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.args.output_dir, 'checkpoint_latest.pt')
        torch.save(checkpoint, latest_path)
        logger.info(f"保存检查点: {latest_path}")
        
        # 保存最佳模型
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'checkpoint_best.pt')
            torch.save(checkpoint, best_path)
            
            # HuggingFace格式
            hf_path = os.path.join(self.args.output_dir, 'best_model')
            self.model.save_pretrained(hf_path)
            logger.info(f"★ 保存最佳模型: {hf_path}")
    
    def load_checkpoint(self, checkpoint_path):
        """加载检查点"""
        if not os.path.exists(checkpoint_path):
            logger.warning(f"检查点不存在: {checkpoint_path}")
            return False
        
        logger.info(f"加载检查点: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"✓ 从step {self.global_step}恢复训练")
        return True
    
    def train(self):
        """完整训练流程"""
        logger.info("=" * 60)
        logger.info("开始CIBD训练")
        logger.info("=" * 60)
        
        for epoch in range(self.args.num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            logger.info(f"\nEpoch {epoch+1}/{self.args.num_epochs}:")
            logger.info(f"  训练损失: {train_loss:.4f}")
            
            # 验证
            if self.val_loader:
                val_loss = self.validate()
                logger.info(f"  验证损失: {val_loss:.4f}")
                
                # 保存最佳
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
                    logger.info(f"  ★ 新的最佳模型! (val_loss: {val_loss:.4f})")
                
                self.save_checkpoint(epoch, is_best)
            else:
                self.save_checkpoint(epoch, is_best=False)
        
        logger.info("\n" + "=" * 60)
        logger.info("训练完成!")
        logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="CIBD训练 - 完整版")
    
    # 模型参数
    parser.add_argument('--teacher_model_path', type=str, required=True,
                       help='教师模型路径')
    parser.add_argument('--compression_ratio', type=float, default=0.5,
                       help='压缩率 (0.3-0.7)')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='训练数据JSON路径')
    parser.add_argument('--val_data_path', type=str, default=None,
                       help='验证数据JSON路径（可选）')
    parser.add_argument('--image_folder', type=str, required=True,
                       help='图像文件夹路径')
    parser.add_argument('--auto_split', action='store_true',
                       help='自动分割训练数据')
    parser.add_argument('--val_split', type=float, default=0.05,
                       help='验证集比例（auto_split时使用）')
    
    # 训练参数
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='批次大小')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='最大序列长度')
    parser.add_argument('--fp16', action='store_true',
                       help='使用混合精度训练')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载workers数')
    
    # Checkpoint
    parser.add_argument('--resume', type=str, default=None,
                       help='从检查点恢复训练')
    
    # 其他
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    
    # 设置随机种子
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    logger.info(f"GPU数量: {torch.cuda.device_count()}")
    
    # 1. 加载教师模型
    logger.info("\n" + "=" * 60)
    logger.info("加载教师模型...")
    logger.info("=" * 60)
    
    tokenizer, teacher_model, image_processor, context_len = load_pretrained_model(
        model_path=args.teacher_model_path,
        model_base=None,
        model_name=os.path.basename(args.teacher_model_path),
        device_map="cpu"
    )
    
    logger.info("\n检查 tokenizer 配置...")
    tokenizer_vocab_size = len(tokenizer)
    model_vocab_size = teacher_model.config.vocab_size

    logger.info(f"  Tokenizer vocab_size: {tokenizer_vocab_size}")
    logger.info(f"  Model vocab_size: {model_vocab_size}")

    if tokenizer_vocab_size != model_vocab_size:
        logger.warning(f"  ⚠️  检测到不匹配!")
        teacher_model.config.vocab_size = tokenizer_vocab_size
        logger.info(f"  ✓ 已更新为: {tokenizer_vocab_size}")

    teacher_model.to(device)
    teacher_model.eval()
    
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    logger.info(f"✓ 教师模型: {teacher_params/1e6:.2f}M 参数")
    
    # 2. 创建学生模型
    logger.info("\n" + "=" * 60)
    logger.info("创建学生模型...")
    logger.info("=" * 60)
    
    teacher_config = teacher_model.config
    student_config = create_compressed_student_config(
        teacher_config,
        compression_ratio=args.compression_ratio
    )
    
    student_model = LlavaCIBDModel(student_config, teacher_model)
    

    # ===== 【添加这段】检查并验证 embedding 大小 =====
    logger.info("\n检查学生模型 embedding 层...")
    if hasattr(student_model.model, 'embed_tokens'):
        actual_embed_vocab = student_model.model.embed_tokens.num_embeddings
        logger.info(f"  embed_tokens.num_embeddings: {actual_embed_vocab}")
        
        if actual_embed_vocab != len(tokenizer):
            logger.warning(f"  ⚠️  Embedding 大小不匹配，进行 resize...")
            student_model.resize_token_embeddings(len(tokenizer))
            logger.info(f"  ✓ Resized to {len(tokenizer)}")
        else:
            logger.info(f"  ✓ Embedding 大小正确")

    if hasattr(student_model, 'lm_head'):
        actual_lmhead_vocab = student_model.lm_head.out_features
        logger.info(f"  lm_head.out_features: {actual_lmhead_vocab}")
        
        if actual_lmhead_vocab != len(tokenizer):
            logger.error(f"  ❌ LM head 大小不匹配!")
            raise ValueError("LM head size mismatch!")
    # ===== 添加结束 =====
    # 初始化权重
    logger.info("从教师模型初始化学生权重...")
    student_model = initialize_student_from_teacher(
        student_model, teacher_model, student_config
    )
    logger.info("="*60)
    logger.info("复制视觉组件（不压缩）...")
    logger.info("="*60)

    # 复制vision tower
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'vision_tower'):
        logger.info("复制vision tower...")
        student_model.model.vision_tower = teacher_model.model.vision_tower
        # 冻结vision tower参数
        for param in student_model.model.vision_tower.parameters():
            param.requires_grad = False
        logger.info("✓ Vision tower已复制并冻结")
    else:
        logger.error("✗ 未找到teacher的vision tower!")
        raise RuntimeError("Cannot find vision_tower in teacher model")

    # 复制mm_projector
    if hasattr(teacher_model, 'model') and hasattr(teacher_model.model, 'mm_projector'):
        logger.info("复制mm_projector...")
        student_model.model.mm_projector = teacher_model.model.mm_projector
        logger.info("✓ MM projector已复制")

    logger.info("="*60)
    student_model.to(device)
    

    # 5. ===== 最终验证（测试 embedding lookup）=====
    logger.info("\n" + "=" * 60)
    logger.info("最终验证...")
    logger.info("=" * 60)
    
    logger.info(f"Tokenizer vocab: {len(tokenizer)}")
    logger.info(f"Teacher embed: {teacher_model.model.embed_tokens.num_embeddings}")
    logger.info(f"Student embed: {student_model.model.embed_tokens.num_embeddings}")
    logger.info(f"Teacher lm_head: {teacher_model.lm_head.out_features}")
    logger.info(f"Student lm_head: {student_model.lm_head.out_features}")
    
    # 测试一个样本
    logger.info("\n测试 embedding lookup...")
    try:
        test_text = "USER: <image>\nWhat is this? ASSISTANT:"
        test_ids = tokenizer(test_text, return_tensors='pt').input_ids
        logger.info(f"  Test IDs shape: {test_ids.shape}")
        logger.info(f"  Test IDs 范围: [{test_ids.min().item()}, {test_ids.max().item()}]")
        
        # 检查是否有超范围的 token
        max_id = test_ids.max().item()
        vocab_size = len(tokenizer)
        if max_id >= vocab_size:
            logger.error(f"  ❌ Token ID {max_id} >= vocab_size {vocab_size}!")
            raise ValueError("Token ID out of range!")
        
        with torch.no_grad():
            test_embed = student_model.model.embed_tokens(test_ids.to(device))
        logger.info(f"  ✓ Embedding 成功! shape: {test_embed.shape}")
    except Exception as e:
        logger.error(f"  ❌ Embedding 失败: {e}")
        import traceback
        traceback.print_exc()
        raise

    # 分析压缩
    analyze_compression(teacher_model, student_model)
    
    # 3. 准备数据
    logger.info("\n" + "=" * 60)
    logger.info("准备数据集...")
    logger.info("=" * 60)
    
    # 处理训练/验证分割
    if args.val_data_path:
        train_json = args.train_data_path
        val_json = args.val_data_path
        logger.info(f"使用提供的验证集")
    elif args.auto_split:
        logger.info(f"自动分割数据 ({args.val_split:.1%} 验证)")
        
        with open(args.train_data_path, 'r') as f:
            all_data = json.load(f)
        
        split_idx = int(len(all_data) * (1 - args.val_split))
        train_data = all_data[:split_idx]
        val_data = all_data[split_idx:]
        
        os.makedirs(args.output_dir, exist_ok=True)
        train_json = os.path.join(args.output_dir, 'train_split.json')
        val_json = os.path.join(args.output_dir, 'val_split.json')
        
        with open(train_json, 'w') as f:
            json.dump(train_data, f)
        with open(val_json, 'w') as f:
            json.dump(val_data, f)
        
        logger.info(f"分割: {len(train_data)} 训练, {len(val_data)} 验证")
    else:
        train_json = args.train_data_path
        val_json = None
        logger.info("仅训练，无验证")
    
    # 创建数据集
    train_dataset = LLaVADataset(
        train_json,
        args.image_folder,
        tokenizer,
        image_processor,
        max_length=args.max_length,
        is_training=True
    )
    
    val_dataset = None
    if val_json:
        val_dataset = LLaVADataset(
            val_json,
            args.image_folder,
            tokenizer,
            image_processor,
            max_length=args.max_length,
            is_training=False
        )
    
    logger.info(f"✓ 训练集: {len(train_dataset)} 样本")
    if val_dataset:
        logger.info(f"✓ 验证集: {len(val_dataset)} 样本")
    
    # 数据加载器
    collator = LLaVACollator(tokenizer, image_processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=args.num_workers,
            pin_memory=True
        )
    
    # 4. 创建训练器
    trainer = CIBDTrainer(student_model, train_loader, val_loader, args, device)
    
    # 恢复训练（如果需要）
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # 5. 开始训练
    trainer.train()
    
    # 6. 保存最终模型
    logger.info("\n" + "=" * 60)
    logger.info("保存最终模型...")
    logger.info("=" * 60)
    
    final_path = os.path.join(args.output_dir, 'final_model')
    student_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    logger.info(f"✓ 最终模型: {final_path}")
    
    # 保存统计信息
    student_params = sum(p.numel() for p in student_model.parameters())
    stats = {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': 1 - (student_params / teacher_params),
        'config': student_config.to_dict()
    }
    
    with open(os.path.join(args.output_dir, 'compression_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info("\n" + "=" * 60)
    logger.info("✓ 训练完成!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
