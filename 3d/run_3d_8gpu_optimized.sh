#!/bin/bash
# Optimized 3D parallel training script for 8x A100 GPUs
# Uses all optimizations except changing model size

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DDP_BACKEND=nccl

# Optimize NCCL settings for A100
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=3
export NCCL_P2P_LEVEL=5

# Set CUDA optimization flags
export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "Starting optimized 3D parallel training..."
torchrun --standalone --nproc_per_node=8 /home/jian.sha/nanoGPT/3d/train_3d_optimized.py 3d/config/train_3d_8gpu_optimized.py
