#!/bin/bash
# Run 3D parallel training on single node with 8 GPUs

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export DDP_BACKEND=nccl

torchrun --standalone --nproc_per_node=8 /home/jian.sha/nanoGPT/3d/train_3d.py 3d/config/train_3d_8gpu.py
