#!/bin/bash

# Stable 3D Parallel Training Script for nanoGPT
# Uses optimized train_3d_optimized.py with stable parameters
# Now supports H20, H100, A100 and other GPUs with automatic MFU calculation

echo "Starting STABLE 3D parallel training with optimized parameters..."

# Set environment variables for stability
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_DEBUG=INFO
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "📋 Stable 3D Parallelism Configuration (built into train_3d_optimized.py):"
echo "  Tensor parallel size: 2 (reduced from 4 for stability)"
echo "  Pipeline parallel size: 1"
echo "  Data parallel size: 4 (8/2/1)"
echo "  Total world size: 8"
echo "  Batch size: 8 (reduced for stability)"
echo "  Block size: 1024 (reduced for stability)"
echo "  Learning rate: 5e-5 (conservative)"
echo "  Gradient clipping: 0.5 (strict)"
echo "  Dropout: 0.1 (increased for regularization)"
echo ""
echo "🔧 MFU Calculation Improvements:"
echo "  - Automatic GPU detection (H20/H100/A100/etc.)"
echo "  - Correct peak FLOPS for each GPU type"
echo "  - Proper tensor parallelism accounting"

# Run with torchrun using the optimized parameters built into the script
torchrun \
    --nproc_per_node=8 \
    --master_port=29500 \
    train_3d_optimized.py
