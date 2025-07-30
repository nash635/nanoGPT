"""
3D Parallel Training Script for nanoGPT with Megatron-LM style parallelism
Supports Data Parallel (DP), Tensor Parallel (TP), and Pipeline Parallel (PP)
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

# Import our 3D parallel components
from megatron.initialize import initialize_model_parallel, destroy_model_parallel
from megatron.initialize import get_data_parallel_group, get_data_parallel_rank, get_data_parallel_world_size
from megatron.model import ParallelGPT, GPTConfig


# -----------------------------------------------------------------------------
# Default config values for 3D parallel training
# I/O
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'nanogpt-3d'
wandb_run_name = 'gpt2-3d'

# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8
batch_size = 12
block_size = 1024

# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# 3D parallelism configuration
tensor_model_parallel_size = 1  # Tensor parallel size
pipeline_model_parallel_size = 1  # Pipeline parallel size
# Data parallel size is computed automatically

# adamw optimizer
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay settings
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# system
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = False  # Disable for 3D parallelism compatibility

# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
# Import config overrides from command line argument
import sys
print(f"Debug: sys.argv = {sys.argv}")
print(f"Debug: Current working directory = {os.getcwd()}")
if len(sys.argv) > 1:
    config_file = sys.argv[1]
    print(f"Debug: Trying to load config from {config_file}")
    if os.path.exists(config_file):
        print(f"Loading config from {config_file}")
        exec(open(config_file).read())
        print(f"Debug: tensor_model_parallel_size = {globals().get('tensor_model_parallel_size', 'NOT SET')}")
    else:
        print(f"Config file {config_file} not found, using defaults")
elif os.path.exists('configurator.py'):
    exec(open('configurator.py').read())

config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------

def setup_distributed():
    """Setup distributed training environment"""
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
        # Use gloo backend for better stability with P2P issues
        backend = os.environ.get('DDP_BACKEND', 'gloo')
        print(f"Initializing distributed with backend: {backend}")
        
        # Set timeout to prevent hanging
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=60)
        )
        torch.cuda.set_device(local_rank)
        print(f"Distributed initialization complete. Rank {rank}/{world_size}")
    
    return rank, local_rank, world_size


def get_batch(split, data_dir):
    """Load a batch of data"""
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    
    # Adjust batch size for data parallel
    dp_world_size = get_data_parallel_world_size()
    dp_rank = get_data_parallel_rank()
    
    # Calculate effective batch size per data parallel rank
    batch_size_per_dp = batch_size // dp_world_size
    if batch_size_per_dp <= 0:
        batch_size_per_dp = 1  # Ensure at least 1 sample per rank
    
    # Debug print for the first call
    if not hasattr(get_batch, '_debug_printed'):
        print(f"get_batch debug: dp_world_size={dp_world_size}, dp_rank={dp_rank}, batch_size_per_dp={batch_size_per_dp}")
        print(f"Data length: {len(data)}, block_size: {block_size}")
        get_batch._debug_printed = True
    
    # Generate random indices with error handling
    try:
        max_start_idx = len(data) - block_size - 1
        if max_start_idx <= 0:
            raise ValueError(f"Data file too small: {len(data)} <= {block_size}")
        
        ix = torch.randint(0, max_start_idx, (batch_size_per_dp,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if device == 'cuda':
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y
        
    except Exception as e:
        print(f"Error in get_batch: {e}")
        print(f"dp_world_size={dp_world_size}, dp_rank={dp_rank}, batch_size_per_dp={batch_size_per_dp}")
        print(f"Data length: {len(data)}, block_size: {block_size}")
        raise


@torch.no_grad()
def estimate_loss(model, data_dir, ctx):
    """Estimate loss on train and val datasets"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


def get_lr(it):
    """Learning rate schedule"""
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)


def main():
    """Main training function"""
    
    # Setup distributed training
    rank, local_rank, world_size = setup_distributed()
    master_process = rank == 0
    
    # Initialize 3D model parallelism
    if master_process:
        print("Initializing 3D model parallelism...")
    initialize_model_parallel(
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size
    )
    if master_process:
        print("3D model parallelism initialized successfully")
    
    # Calculate tokens per iteration considering 3D parallelism
    data_parallel_size = get_data_parallel_world_size()
    tokens_per_iter = gradient_accumulation_steps * data_parallel_size * batch_size * block_size
    
    if master_process:
        print(f"3D Parallelism Configuration:")
        print(f"  Tensor parallel size: {tensor_model_parallel_size}")
        print(f"  Pipeline parallel size: {pipeline_model_parallel_size}") 
        print(f"  Data parallel size: {data_parallel_size}")
        print(f"  Total world size: {world_size}")
        print(f"Tokens per iteration: {tokens_per_iter:,}")
        os.makedirs(out_dir, exist_ok=True)
    
    # Set random seed
    torch.manual_seed(1337 + rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    device_type = 'cuda' if 'cuda' in device else 'cpu'
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    
    # Data loading
    data_dir = os.path.join('..', 'data', dataset)  # Correct path from 3d directory
    if not os.path.exists(os.path.join(data_dir, 'train.bin')):
        print(f"Data directory {data_dir} not found, trying alternative path...")
        data_dir = os.path.join('data', dataset)
        if not os.path.exists(os.path.join(data_dir, 'train.bin')):
            raise FileNotFoundError(f"Training data not found in {data_dir}")
    
    # Model initialization
    gptconf = GPTConfig(
        block_size=block_size,
        vocab_size=50304,  # GPT-2 vocab size
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        tensor_model_parallel_size=tensor_model_parallel_size,
        pipeline_model_parallel_size=pipeline_model_parallel_size
    )
    
    if init_from == 'scratch':
        if master_process:
            print("Initializing a new model from scratch")
        model = ParallelGPT(gptconf)
    else:
        raise NotImplementedError("Resume and pretrained not implemented for 3D parallel model")
    
    # Move model to device
    model.to(device)
    
    # Initialize gradient scaler
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    
    # Optimizer (configure before DDP wrapping)
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # Wrap with DDP for data parallelism after optimizer configuration
    if data_parallel_size > 1:
        model = DDP(model, process_group=get_data_parallel_group())
    
    # Compile model (disabled for now due to compatibility)
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model)
    
    # Training loop variables
    iter_num = 0
    best_val_loss = 1e9
    
    # First evaluation
    if eval_only:
        losses = estimate_loss(model, data_dir, ctx)
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        return
    
    # Training loop
    X, Y = get_batch('train', data_dir)  # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0
    running_mfu = -1.0
    
    while True:
        # Determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # Evaluate the loss on train/val sets and write checkpoints
        if iter_num % eval_interval == 0 and master_process and iter_num > 0:  # Skip eval at iter_num=0
            losses = estimate_loss(model, data_dir, ctx)
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            
            if losses['val'] < best_val_loss or always_save_checkpoint:
                best_val_loss = losses['val']
                if iter_num > 0:
                    # Get the underlying model for checkpoint saving
                    model_to_save = model.module if hasattr(model, 'module') else model
                    checkpoint = {
                        'model': model_to_save.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': gptconf,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                        'config': config,
                    }
                    print(f"saving checkpoint to {out_dir}")
                    torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
        
        if iter_num == 0 and eval_only:
            break
        
        # Forward backward update, with optional gradient accumulation to simulate larger batch size
        for micro_step in range(gradient_accumulation_steps):
            with ctx:
                logits, loss = model(X, Y)
                loss = loss / gradient_accumulation_steps
            
            # Immediately async prefetch next batch while model is doing the forward pass on the GPU
            X, Y = get_batch('train', data_dir)
            
            # Backward pass
            scaler.scale(loss).backward()
        
        # Clip the gradient
        if grad_clip != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            # Ensure unscale is called even without clipping to avoid scaler errors
            scaler.unscale_(optimizer)
        
        # Step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        
        # Flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        
        # Timing and logging
        t1 = time.time()
        dt = t1 - t0
        t0 = t1
        if iter_num % log_interval == 0 and master_process:
            # Get loss as float. Note: this is a CPU-GPU sync point
            # Scale back to the original loss (multiply by gradient_accumulation_steps)
            lossf = loss.item() * gradient_accumulation_steps
            if local_iter_num >= 5:  # Let the training loop settle a bit
                # Get the underlying model for MFU calculation
                model_for_mfu = model.module if hasattr(model, 'module') else model
                mfu = model_for_mfu.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
                running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
        # Progress check: print a simple indicator every few iterations  
        if master_process and iter_num > 0 and iter_num % 5 == 0:
            print(f"[INFO] Completed iteration {iter_num}, starting next...")
        
        iter_num += 1
        local_iter_num += 1
        
        # Termination conditions
        if iter_num > max_iters:
            break
    
    # Cleanup
    if world_size > 1:
        destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
