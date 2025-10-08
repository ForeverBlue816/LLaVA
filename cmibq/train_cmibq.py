#!/usr/bin/env python
"""
CM-IBQ LLaVA训练脚本
支持两阶段训练：
  Stage 1: Bottleneck Shaping (量化模块训练)
  Stage 2: Task-Aware Optimization with Alignment (对齐损失 + LoRA微调)
"""

import os
import sys
import argparse
import torch
import random
import numpy as np
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from cmibq import CMIBQQuantizedLLaVA
from cmibq.training import DistributedCMIBQTrainer
from cmibq.training.data_utils import make_supervised_data_module


def set_seed(seed: int):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='CM-IBQ LLaVA Training')
    
    # 模型参数
    parser.add_argument('--model_path', type=str, required=True,
                       help='LLaVA模型路径')
    parser.add_argument('--model_base', type=str, default=None,
                       help='基础模型路径（如果使用LoRA）')
    parser.add_argument('--model_size', type=str, default='7b',
                       choices=['7b', '13b', '34b', '70b'],
                       help='模型大小')
    
    # 量化参数
    parser.add_argument('--target_bits_act', type=float, default=4.0,
                       help='激活量化目标比特')
    parser.add_argument('--target_bits_weight', type=float, default=4.0,
                       help='权重量化目标比特')
    parser.add_argument('--num_groups', type=int, default=8,
                       help='量化分组数')
    parser.add_argument('--use_ib', action='store_true', default=True,
                       help='是否使用IB框架')
    parser.add_argument('--use_lora', action='store_true', default=True,
                       help='是否使用LoRA (Stage 2)')
    parser.add_argument('--lora_rank', type=int, default=16,
                       help='LoRA秩')
    
    # 训练阶段
    parser.add_argument('--stage', type=int, default=1, choices=[1, 2],
                       help='训练阶段: 1=Bottleneck Shaping, 2=Alignment')
    
    # 数据参数
    parser.add_argument('--train_data_path', type=str, required=True,
                       help='训练数据JSON路径')
    parser.add_argument('--eval_data_path', type=str, default=None,
                       help='验证数据JSON路径')
    parser.add_argument('--image_folder', type=str, required=True,
                       help='图像文件夹路径')
    parser.add_argument('--max_length', type=int, default=2048,
                       help='最大序列长度')
    
    # 训练参数
    parser.add_argument('--output_dir', type=str, default='./checkpoints',
                       help='输出目录')
    parser.add_argument('--num_epochs', type=int, default=3,
                       help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='每GPU的batch size')
    parser.add_argument('--eval_batch_size', type=int, default=4,
                       help='验证batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4,
                       help='梯度累积步数')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help='学习率')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='权重衰减')
    parser.add_argument('--warmup_steps', type=int, default=500,
                       help='warmup步数')
    parser.add_argument('--max_grad_norm', type=float, default=1.0,
                       help='梯度裁剪')
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子')
    
    # 优化参数
    parser.add_argument('--use_amp', action='store_true', default=True,
                       help='使用混合精度训练')
    parser.add_argument('--fp16', action='store_true', default=False,
                       help='使用FP16')
    parser.add_argument('--bf16', action='store_true', default=False,
                       help='使用BF16')
    parser.add_argument('--use_deepspeed', action='store_true',
                       help='使用DeepSpeed')
    parser.add_argument('--gradient_checkpointing', action='store_true',
                       help='使用梯度检查点')
    
    # 日志和保存
    parser.add_argument('--logging_steps', type=int, default=10,
                       help='日志记录步数')
    parser.add_argument('--save_epochs', type=int, default=1,
                       help='保存检查点的epoch间隔')
    parser.add_argument('--eval_epochs', type=int, default=1,
                       help='验证的epoch间隔')
    parser.add_argument('--use_wandb', action='store_true',
                       help='使用W&B记录')
    parser.add_argument('--run_name', type=str, default=None,
                       help='运行名称')
    
    # 其他
    parser.add_argument('--num_workers', type=int, default=4,
                       help='数据加载worker数')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None,
                       help='从检查点恢复')
    
    args = parser.parse_args()
    return args


def main():
    """主函数"""
    # 解析参数
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 打印配置
    print("=" * 80)
    print(f"CM-IBQ LLaVA Training - Stage {args.stage}")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Target bits (act/weight): {args.target_bits_act}/{args.target_bits_weight}")
    print(f"Use IB: {args.use_ib}")
    print(f"Use LoRA: {args.use_lora and args.stage == 2}")
    print(f"Output: {args.output_dir}")
    print("=" * 80)
    
    # 1. 加载模型
    print("\n[1/4] Loading model...")
    model = CMIBQQuantizedLLaVA(
        model_path=args.model_path,
        target_bits_act=args.target_bits_act,
        target_bits_weight=args.target_bits_weight,
        use_ib=args.use_ib,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        num_groups=args.num_groups,
        stage=args.stage,
        model_base=args.model_base
    )
    
    print(f"Model loaded successfully!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")
    
    # 2. 准备数据
    print("\n[2/4] Preparing data...")
    data_module = make_supervised_data_module(
        tokenizer=model.tokenizer,
        image_processor=model.image_processor,
        data_args={
            'train_data_path': args.train_data_path,
            'eval_data_path': args.eval_data_path,
            'image_folder': args.image_folder,
            'max_length': args.max_length
        }
    )
    
    train_dataset = data_module['train_dataset']
    eval_dataset = data_module['eval_dataset']
    
    print(f"Train samples: {len(train_dataset) if train_dataset else 0}")
    print(f"Eval samples: {len(eval_dataset) if eval_dataset else 0}")
    
    # 3. 配置训练参数
    print("\n[3/4] Configuring trainer...")
    
    # 自动设置run_name
    if args.run_name is None:
        args.run_name = f"cmibq_stage{args.stage}_{args.target_bits_act}bit"
    
    training_args = {
        # 基础参数
        'output_dir': args.output_dir,
        'num_epochs': args.num_epochs,
        'batch_size': args.batch_size,
        'eval_batch_size': args.eval_batch_size,
        'gradient_accumulation_steps': args.gradient_accumulation_steps,
        
        # 优化参数
        'learning_rate': args.learning_rate,
        'weight_decay': args.weight_decay,
        'warmup_steps': args.warmup_steps,
        'max_grad_norm': args.max_grad_norm,
        
        # 精度设置
        'use_amp': args.use_amp,
        'fp16': args.fp16,
        'bf16': args.bf16,
        'use_deepspeed': args.use_deepspeed,
        'gradient_checkpointing': args.gradient_checkpointing,
        
        # 日志和保存
        'logging_steps': args.logging_steps,
        'save_epochs': args.save_epochs,
        'eval_epochs': args.eval_epochs,
        'use_wandb': args.use_wandb,
        'run_name': args.run_name,
        
        # 其他
        'num_workers': args.num_workers,
        'seed': args.seed,
        'model_size': args.model_size,
        'stage': args.stage,
        
        # DDP设置
        'find_unused_parameters': True,
        'static_graph': False
    }
    
    # 4. 创建训练器
    trainer = DistributedCMIBQTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=model.tokenizer
    )
    
    # 从检查点恢复
    if args.resume_from_checkpoint:
        print(f"\nResuming from checkpoint: {args.resume_from_checkpoint}")
        trainer.load_checkpoint(args.resume_from_checkpoint)
    
    # 5. 开始训练
    print("\n[4/4] Starting training...")
    print("=" * 80)
    
    trainer.train()
    
    print("\n" + "=" * 80)
    print("Training completed!")
    print(f"Best model saved to: {os.path.join(args.output_dir, 'best_model')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
