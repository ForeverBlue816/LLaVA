#!/usr/bin/env python3
"""
修正后的CIBD训练脚本
真正实现模型压缩
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup
import argparse
from tqdm import tqdm

from llava.model.builder import load_pretrained_model
from cibd.llava_cibd_model import LlavaCIBD
from cibd.dataset import LLaVADataset, LLaVACollator
from cibd.create_student_model import create_compressed_student_config, initialize_student_from_teacher


class CIBDTrainer:
    def __init__(
        self,
        teacher_model,
        student_model,
        train_loader,
        val_loader,
        args
    ):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.args = args
        
        # 将模型移到GPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.student_model.to(self.device)
        self.teacher_model.to(self.device)
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
        
        # 学习率调度
        total_steps = len(train_loader) * args.num_epochs
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(total_steps * args.warmup_ratio),
            num_training_steps=total_steps
        )
        
        # 混合精度训练
        self.scaler = torch.cuda.amp.GradScaler() if args.fp16 else None
        
        # 最佳模型跟踪
        self.best_val_loss = float('inf')
        
    def train_step(self, batch):
        """单步训练"""
        self.student_model.train()
        
        # 准备数据
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                for k, v in batch.items()}
        
        # 混合精度上下文
        with torch.cuda.amp.autocast(enabled=self.args.fp16):
            outputs = self.student_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                images=batch['images'],
                labels=batch['labels']
            )
            
            loss = outputs.loss / self.args.gradient_accumulation_steps
        
        # 反向传播
        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        
        return loss.item() * self.args.gradient_accumulation_steps
    
    def train_epoch(self, epoch):
        """训练一个epoch"""
        total_loss = 0
        accumulation_steps = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.args.num_epochs}")
        
        for step, batch in enumerate(progress_bar):
            loss = self.train_step(batch)
            total_loss += loss
            accumulation_steps += 1
            
            # 梯度累积
            if (step + 1) % self.args.gradient_accumulation_steps == 0:
                # 梯度裁剪
                if self.scaler:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                
                # 优化器步进
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                # 更新进度条
                current_lr = self.scheduler.get_last_lr()[0]
                progress_bar.set_postfix({
                    'loss': loss,
                    'lr': current_lr
                })
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """验证模型"""
        self.student_model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
                
                with torch.cuda.amp.autocast(enabled=self.args.fp16):
                    outputs = self.student_model(
                        input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'],
                        images=batch['images'],
                        labels=batch['labels']
                    )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'args': self.args
        }
        
        checkpoint_path = os.path.join(self.args.output_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if is_best:
            best_path = os.path.join(self.args.output_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            
            # 保存为HuggingFace格式
            hf_path = os.path.join(self.args.output_dir, 'hf_model')
            self.student_model.save_pretrained(hf_path)
    
    def train(self):
        """完整训练流程"""
        print("Starting CIBD Training...")
        print(f"Teacher params: {sum(p.numel() for p in self.teacher_model.parameters())/1e6:.2f}M")
        print(f"Student params: {sum(p.numel() for p in self.student_model.parameters())/1e6:.2f}M")
        
        for epoch in range(self.args.num_epochs):
            # 训练
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}")
            
            # 验证
            val_loss = self.validate()
            print(f"Epoch {epoch+1}: Val Loss = {val_loss:.4f}")
            
            # 保存模型
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                print(f"New best model! Val Loss: {val_loss:.4f}")
            
            self.save_checkpoint(epoch, is_best)
        
        print("Training completed!")


def main():
    parser = argparse.ArgumentParser()
    
    # 模型参数
    parser.add_argument('--teacher_model_path', type=str, required=True)
    parser.add_argument('--compression_ratio', type=float, default=0.5)
    
    # 数据参数
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--image_folder', type=str, required=True)
    
    # 训练参数
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--fp16', action='store_true')
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. 加载教师模型
    print("Loading teacher model...")
    tokenizer, teacher_model, image_processor, _ = load_pretrained_model(
        args.teacher_model_path,
        None,
        "teacher",
        device_map="cpu"  # 先在CPU加载
    )
    
    # 2. 创建学生模型配置（真正更小的模型）
    print("Creating student model configuration...")
    teacher_config = teacher_model.config
    student_config = create_compressed_student_config(
        teacher_config, 
        compression_ratio=args.compression_ratio
    )
    
    # 3. 创建学生模型
    print("Creating student model...")
    student_model = LlavaCIBD(student_config, teacher_model)
    
    # 4. 初始化学生模型权重（从教师模型）
    print("Initializing student from teacher...")
    student_model = initialize_student_from_teacher(
        student_model, teacher_model, student_config
    )
    
    # 打印压缩统计
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    actual_compression = 1 - (student_params / teacher_params)
    print(f"Teacher parameters: {teacher_params/1e6:.2f}M")
    print(f"Student parameters: {student_params/1e6:.2f}M")
    print(f"Actual compression ratio: {actual_compression:.2%}")
    
    # 5. 准备数据
    print("Preparing datasets...")
    
    # 这里需要先分割数据
    import json
    with open(args.data_path, 'r') as f:
        all_data = json.load(f)
    
    # 简单分割：90%训练，10%验证
    split_idx = int(len(all_data) * 0.9)
    train_data = all_data[:split_idx]
    val_data = all_data[split_idx:]
    
    # 保存分割后的数据
    train_json = os.path.join(args.output_dir, 'train.json')
    val_json = os.path.join(args.output_dir, 'val.json')
    
    with open(train_json, 'w') as f:
        json.dump(train_data, f)
    with open(val_json, 'w') as f:
        json.dump(val_data, f)
    
    # 创建数据集
    train_dataset = LLaVADataset(
        train_json,
        args.image_folder,
        tokenizer,
        image_processor,
        is_training=True
    )
    
    val_dataset = LLaVADataset(
        val_json,
        args.image_folder,
        tokenizer,
        image_processor,
        is_training=False
    )
    
    # 创建数据加载器
    collator = LLaVACollator(tokenizer, image_processor)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=4
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=4
    )
    
    # 6. 创建训练器并开始训练
    trainer = CIBDTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        train_loader=train_loader,
        val_loader=val_loader,
        args=args
    )
    
    trainer.train()
    
    # 7. 导出最终模型
    print("Exporting final model...")
    final_path = os.path.join(args.output_dir, 'final_compressed_model')
    student_model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    
    # 保存压缩统计
    stats = {
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': actual_compression,
        'student_config': student_config.to_dict()
    }
    
    with open(os.path.join(args.output_dir, 'compression_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Training completed! Final model saved to {final_path}")


if __name__ == "__main__":
    main()
