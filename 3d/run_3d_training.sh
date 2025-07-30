#!/bin/bash

# 3D Parallel nanoGPT Training Script
# Usage: ./run_3d_training.sh [config_file] [num_gpus]

set -e

# Default values
CONFIG_FILE=${1:-"config/train_3d_2gpu.py"}
NUM_GPUS=${2:-2}

echo "Starting 3D Parallel nanoGPT Training"
echo "Configuration: $CONFIG_FILE"
echo "Number of GPUs: $NUM_GPUS"
echo "======================================"

cd /home/jian.sha/nanoGPT/3d

# Check if data exists
if [ ! -f "../data/openwebtext/train.bin" ]; then
    echo "Error: Training data not found at ../data/openwebtext/train.bin"
    exit 1
fi

# Set environment variables
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NUM_GPUS-1)))
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand for compatibility
export NCCL_IGNORE_DISABLED_P2P=1  # Ignore P2P issues between GPUs
export NCCL_P2P_DISABLE=1  # Completely disable P2P
export NCCL_SOCKET_IFNAME=lo  # Use loopback interface
export NCCL_BLOCKING_WAIT=1  # Use blocking wait for stability
export NCCL_TREE_THRESHOLD=0  # Force tree algorithm for small sizes

# Additional debug settings
export TORCH_DISTRIBUTED_DEBUG=INFO

# Run training
echo "Launching training with torchrun..."
torchrun \
    --standalone \
    --nproc_per_node=$NUM_GPUS \
    train_3d.py \
    $CONFIG_FILE

echo "Training completed or interrupted."
