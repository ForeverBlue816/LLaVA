#!/bin/bash
# ============================================================================
# CM-IBQ LLaVA 训练脚本集合
# ============================================================================

# ----------------------------------------------------------------------------
# 1. 单机单卡训练 - Stage 1 (Bottleneck Shaping)
# ----------------------------------------------------------------------------
cat > train_stage1_single_gpu.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_4bit" \
    --num_epochs 3 \
    --batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_single_gpu.sh


# ----------------------------------------------------------------------------
# 2. 单机多卡训练 - Stage 1 (使用DDP)
# ----------------------------------------------------------------------------
cat > train_stage1_multi_gpu.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

NUM_GPUS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_4bit_4gpu" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_multi_gpu.sh


# ----------------------------------------------------------------------------
# 3. 单机多卡训练 - Stage 1 (使用DeepSpeed ZeRO-2)
# ----------------------------------------------------------------------------
cat > train_stage1_deepspeed.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

NUM_GPUS=4

deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_4bit_deepspeed" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_deepspeed \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_deepspeed.sh


# ----------------------------------------------------------------------------
# 4. Stage 2训练 (Alignment + LoRA) - 多卡
# ----------------------------------------------------------------------------
cat > train_stage2_multi_gpu.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

NUM_GPUS=4

# 注意：需要先运行Stage 1得到检查点
STAGE1_CHECKPOINT="./checkpoints/stage1_4bit_4gpu/best_model"

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "$STAGE1_CHECKPOINT" \
    --model_size "7b" \
    \
    --stage 2 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    --use_lora \
    --lora_rank 16 \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage2_4bit_aligned_4gpu" \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --weight_decay 0.01 \
    --warmup_steps 300 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage2_multi_gpu.sh


# ----------------------------------------------------------------------------
# 5. 多机多卡训练 - Stage 1
# ----------------------------------------------------------------------------
cat > train_stage1_multi_node.sh << 'EOF'
#!/bin/bash

# 节点配置
export MASTER_ADDR="主节点IP"
export MASTER_PORT=29500
export NNODES=2  # 节点数
export NODE_RANK=0  # 当前节点序号，主节点为0
export GPUS_PER_NODE=8

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=8

torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-13b" \
    --model_size "13b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_13b_multinode" \
    --num_epochs 3 \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --warmup_steps 500 \
    --max_grad_norm 1.0 \
    \
    --use_amp \
    --bf16 \
    --gradient_checkpointing \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_stage1_multi_node.sh


# ----------------------------------------------------------------------------
# 6. W&B监控训练脚本
# ----------------------------------------------------------------------------
cat > train_with_wandb.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# 设置W&B
export WANDB_PROJECT="cm-ibq-llava"
export WANDB_API_KEY="你的API密钥"

NUM_GPUS=4

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    \
    --output_dir "./checkpoints/stage1_wandb" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    \
    --use_amp \
    --fp16 \
    --gradient_checkpointing \
    \
    --use_wandb \
    --run_name "cmibq_stage1_4bit_experiment1" \
    \
    --logging_steps 10 \
    --save_epochs 1 \
    --eval_epochs 1 \
    --num_workers 4 \
    --seed 42
EOF

chmod +x train_with_wandb.sh


# ----------------------------------------------------------------------------
# 7. 快速测试脚本（小数据集）
# ----------------------------------------------------------------------------
cat > train_quick_test.sh << 'EOF'
#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

python train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    \
    --train_data_path "./data/llava_instruct_small.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 512 \
    \
    --output_dir "./checkpoints/quick_test" \
    --num_epochs 1 \
    --batch_size 1 \
    --gradient_accumulation_steps 2 \
    --learning_rate 2e-5 \
    \
    --use_amp \
    --fp16 \
    \
    --logging_steps 5 \
    --save_epochs 1 \
    --num_workers 2 \
    --seed 42
EOF

chmod +x train_quick_test.sh


# ----------------------------------------------------------------------------
# 8. 完整的两阶段训练pipeline
# ----------------------------------------------------------------------------
cat > train_full_pipeline.sh << 'EOF'
#!/bin/bash

set -e  # 遇到错误就退出

echo "=========================================="
echo "CM-IBQ Full Training Pipeline"
echo "=========================================="

# 配置
NUM_GPUS=4
export CUDA_VISIBLE_DEVICES=0,1,2,3
export OMP_NUM_THREADS=8

# -------------------- Stage 1 --------------------
echo ""
echo "[Stage 1/2] Bottleneck Shaping Training..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "liuhaotian/llava-v1.5-7b" \
    --model_size "7b" \
    --stage 1 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    --output_dir "./checkpoints/stage1_final" \
    --num_epochs 3 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 2e-5 \
    --use_amp --fp16 --gradient_checkpointing \
    --logging_steps 10 --save_epochs 1 --eval_epochs 1 \
    --num_workers 4 --seed 42

echo ""
echo "Stage 1 completed!"
echo ""

# -------------------- Stage 2 --------------------
echo "[Stage 2/2] Task-Aware Optimization with Alignment..."
echo ""

torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_cmibq.py \
    --model_path "./checkpoints/stage1_final/best_model" \
    --model_size "7b" \
    --stage 2 \
    --target_bits_act 4.0 \
    --target_bits_weight 4.0 \
    --num_groups 8 \
    --use_ib \
    --use_lora \
    --lora_rank 16 \
    --train_data_path "./data/llava_instruct_150k.json" \
    --eval_data_path "./data/llava_instruct_eval.json" \
    --image_folder "./data/coco/train2017" \
    --max_length 2048 \
    --output_dir "./checkpoints/stage2_final" \
    --num_epochs 2 \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 1e-5 \
    --use_amp --fp16 --gradient_checkpointing \
    --logging_steps 10 --save_epochs 1 --eval_epochs 1 \
    --num_workers 4 --seed 42

echo ""
echo "=========================================="
echo "Full pipeline completed!"
echo "Final model: ./checkpoints/stage2_final/best_model"
echo "=========================================="
EOF

chmod +x train_full_pipeline.sh


echo "All training scripts created successfully!"
echo ""
echo "Available scripts:"
echo "  1. train_stage1_single_gpu.sh      - 单卡训练 Stage 1"
echo "  2. train_stage1_multi_gpu.sh       - 多卡DDP训练 Stage 1"
echo "  3. train_stage1_deepspeed.sh       - DeepSpeed训练 Stage 1"
echo "  4. train_stage2_multi_gpu.sh       - 多卡训练 Stage 2"
echo "  5. train_stage1_multi_node.sh      - 多机训练 Stage 1"
echo "  6. train_with_wandb.sh             - 带W&B监控的训练"
echo "  7. train_quick_test.sh             - 快速测试"
echo "  8. train_full_pipeline.sh          - 完整两阶段流程"
