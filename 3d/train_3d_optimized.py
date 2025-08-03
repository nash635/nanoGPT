"""
Optimized 3D Parallel Training Script for high GPU utilization
Enhanced version with performance optimizations for A100 GPUs
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
from datetime import timedelta
import numpy as np

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Import our 3D parallel components
from megatron.initialize import initialize_model_parallel, destroy_model_parallel
from megatron.initialize import get_data_parallel_group, get_data_parallel_rank, get_data_parallel_world_size
from megatron.model import ParallelGPT, GPTConfig


# -----------------------------------------------------------------------------
# Default config values for optimized 3D parallel training
# I/O
out_dir = 'out'
eval_interval = 500
log_interval = 1
eval_iters = 100
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'nanogpt-3d-optimized'
wandb_run_name = 'gpt2-3d-optimized'

# tensorboard logging
tensorboard_log = True
tensorboard_log_dir = 'runs/3d_parallel_optimized'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 32  # Increased for better GPU utilization
batch_size = 16  # Increased batch size
block_size = 2048  # Increased sequence length for A100

# model - Keep original GPT-2 124M size
n_layer = 12  # Original GPT-2 small
n_head = 12   # Original GPT-2 small
n_embd = 768  # Original GPT-2 small
dropout = 0.0 # Original setting
bias = False
base_vocab_size = 50257

# 3D parallelism configuration - Optimized for 8x A100
tensor_model_parallel_size = 4  # 4-way tensor parallelism
pipeline_model_parallel_size = 1  # Disable pipeline to reduce overhead
# Data parallel size is computed automatically: 8/(4*1) = 2

# adamw optimizer
learning_rate = 3e-4  # Adjusted for larger model
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 3e-5

# system
device = 'cuda'
dtype = 'bfloat16'  # Use bfloat16 for A100
compile = True  # Enable compilation for better performance

# Performance optimizations
pin_memory = True
non_blocking = True
prefetch_factor = 4  # Increased prefetching

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# Import config overrides from command line argument
import sys
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    if os.path.exists(config_file):
        exec(open(config_file).read())
elif os.path.exists('configurator.py'):
    exec(open('configurator.py').read())

config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------

def setup_distributed():
    """Setup distributed training environment with optimizations"""
    # Initialize distributed backend
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Not using distributed training")
        rank = 0
        local_rank = 0
        world_size = 1
    
    if world_size > 1:
        # Use nccl backend for GPU communication
        backend = os.environ.get('DDP_BACKEND', 'nccl')
        
        # Set timeout to prevent hanging
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=180)  # Increased timeout
        )
        torch.cuda.set_device(local_rank)
        
        # Performance optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True  # Optimize cudnn for consistent input sizes
    
    return rank, local_rank, world_size


class OptimizedDataLoader:
    """Optimized data loader with prefetching and memory pinning"""
    
    def __init__(self, data_dir, split, block_size, batch_size_per_dp, device):
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.batch_size_per_dp = batch_size_per_dp
        self.device = device
        
        # Load data
        if split == 'train':
            self.data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            self.data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        
        self.max_start_idx = len(self.data) - block_size - 1
        if self.max_start_idx <= 0:
            raise ValueError(f"Data file too small: {len(self.data)} <= {block_size}")
    
    def get_batch(self):
        """Generate a batch with optimized memory operations"""
        ix = torch.randint(0, self.max_start_idx, (self.batch_size_per_dp,))
        
        # Use more efficient tensor operations
        x = torch.empty((self.batch_size_per_dp, self.block_size), dtype=torch.long)
        y = torch.empty((self.batch_size_per_dp, self.block_size), dtype=torch.long)
        
        for i, start_idx in enumerate(ix):
            x[i] = torch.from_numpy(self.data[start_idx:start_idx+self.block_size].astype(np.int64))
            y[i] = torch.from_numpy(self.data[start_idx+1:start_idx+1+self.block_size].astype(np.int64))
        
        # Clamp target values
        y = torch.clamp(y, 0, base_vocab_size - 1)
        
        if self.device == 'cuda':
            if pin_memory:
                x = x.pin_memory()
                y = y.pin_memory()
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y


def get_batch(split, data_dir):
    """Generate a batch of data - optimized version"""
    dp_world_size = get_data_parallel_world_size()
    dp_rank = get_data_parallel_rank()
    batch_size_per_dp = batch_size // dp_world_size
    
    # Use optimized data loader
    if not hasattr(get_batch, 'loaders'):
        get_batch.loaders = {}
    
    loader_key = f"{split}_{data_dir}"
    if loader_key not in get_batch.loaders:
        get_batch.loaders[loader_key] = OptimizedDataLoader(
            data_dir, split, block_size, batch_size_per_dp, device
        )
    
    return get_batch.loaders[loader_key].get_batch()


@torch.no_grad()
def estimate_loss(model, data_dir, ctx):
    """Estimate loss on train and val datasets - optimized"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)  # Keep on GPU
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it):
    """Learning rate schedule"""
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    if it > lr_decay_iters:
        return min_lr
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    """Main training function with optimizations"""
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    master_process = rank == 0
    
    # Initialize 3D model parallelism
    initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size
    )
    
    # Calculate tokens per iteration considering 3D parallelism
    data_parallel_size = get_data_parallel_world_size()
    tokens_per_iter = gradient_accumulation_steps * data_parallel_size * batch_size * block_size
    
    if master_process:
        print(f"🚀 Optimized 3D Parallelism Configuration:")
        print(f"  Tensor parallel size: {tensor_model_parallel_size}")
        print(f"  Pipeline parallel size: {pipeline_model_parallel_size}") 
        print(f"  Data parallel size: {data_parallel_size}")
        print(f"  Total world size: {world_size}")
        print(f"⚡ Performance Settings:")
        print(f"  Tokens per iteration: {tokens_per_iter:,}")
        print(f"  Batch size: {batch_size}")
        print(f"  Block size: {block_size}")
        print(f"  Gradient accumulation steps: {gradient_accumulation_steps}")
        print(f"  Model parameters: ~{(n_layer * n_head * n_embd * 12) / 1e6:.0f}M (GPT-2 124M)")
        os.makedirs(out_dir, exist_ok=True)
        
        # Initialize TensorBoard writer
        if tensorboard_log:
            tb_log_dir = os.path.join(tensorboard_log_dir, f"run_{time.strftime('%Y%m%d_%H%M%S')}")
            os.makedirs(tb_log_dir, exist_ok=True)
            writer = SummaryWriter(tb_log_dir)
            print(f"📊 TensorBoard logs will be saved to: {tb_log_dir}")
        else:
            writer = None
    else:
        writer = None
    
    # Set random seed
    torch.manual_seed(1337 + rank)
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Data loading
    data_dir = os.path.join('..', 'data', dataset)
    if not os.path.exists(os.path.join(data_dir, 'train.bin')):
        data_dir = os.path.join('data', dataset)
        if not os.path.exists(os.path.join(data_dir, 'train.bin')):
            data_dir = '/home/jian.sha/nanoGPT/data/openwebtext'
            if not os.path.exists(os.path.join(data_dir, 'train.bin')):
                raise FileNotFoundError(f"Training data not found in {data_dir}")
    
    # Model initialization with optimized vocabulary size
    vocab_size = ((base_vocab_size - 1) // tensor_model_parallel_size + 1) * tensor_model_parallel_size
    
    gptconf = GPTConfig(
        block_size=block_size,
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size
    )
    
    if master_process:
        print(f"🔧 Using vocab_size={vocab_size} (rounded up from {base_vocab_size} for tensor parallelism)")
    
    if init_from == 'scratch':
        model = ParallelGPT(gptconf)
    else:
        raise NotImplementedError("Resume and pretrained not implemented for 3D parallel model")
    
    # Move model to device
    model.to(device)
    
    # Initialize gradient scaler with optimizations
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    
    # Optimizer with optimizations
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # Wrap with DDP for data parallelism
    if data_parallel_size > 1:
        model = DDP(
            model, 
            process_group=get_data_parallel_group(),
            find_unused_parameters=False,  # Optimization for static graphs
            broadcast_buffers=False  # Don't sync buffers if not needed
        )
    
    # Compile model for better performance
    if compile:
        if master_process:
            print("🔥 Compiling model for better performance...")
        unoptimized_model = model
        model = torch.compile(model)
    
    # Training loop variables
    iter_num = 0
    best_val_loss = 1e9
    
    if eval_only:
        losses = estimate_loss(model, data_dir, ctx)
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        return
    
    # Pre-fetch first batch
    X, Y = get_batch('train', data_dir)
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    if master_process:
        print("🎯 Starting optimized training loop...")
    
    while True:
        # Learning rate scheduling
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluation and checkpointing
        if iter_num % eval_interval == 0 and master_process and iter_num > 0:
            losses = estimate_loss(model, data_dir, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if writer is not None:
                writer.add_scalar('Loss/Train', losses['train'], iter_num)
                writer.add_scalar('Loss/Validation', losses['val'], iter_num)
                writer.add_scalar('Learning_Rate', lr, iter_num)
            
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    model_to_save = model.module if hasattr(model, 'module') else model
                    checkpoint = {
                        'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': gptconf,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"💾 saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        if iter_num == 0 and eval_only:
            break
        
        # Forward backward update with gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            
            # Prefetch next batch asynchronously
            X, Y = get_batch('train', data_dir)
            
            # Backward pass
            scaler.scale(loss).backward()
        
        # Gradient clipping and optimizer step
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            scaler.unscale_(optimizer)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        
        if iter_num % log_interval == 0 and master_process:
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:
                model_for_mfu = model.module if hasattr(model, 'module') else model
                mfu = model_for_mfu.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            
            # Enhanced logging
            tokens_per_sec = tokens_per_iter / dt
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%, tok/s {tokens_per_sec:.0f}")
            
            if writer is not None:
                writer.add_scalar('Loss/Train_Step', lossf, iter_num)
                writer.add_scalar('Performance/Time_per_Step_ms', dt*1000, iter_num)
                writer.add_scalar('Performance/Tokens_per_Second', tokens_per_sec, iter_num)
                if running_mfu > 0:
                    writer.add_scalar('Performance/MFU_Percent', running_mfu*100, iter_num)
        
        iter_num += 1
        local_iter_num += 1
        
        if iter_num > max_iters:
            break
    
    # Cleanup
    if master_process and writer is not None:
        writer.close()
        print(f"📊 TensorBoard logs saved to: {tb_log_dir}")
    
    if world_size > 1:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
