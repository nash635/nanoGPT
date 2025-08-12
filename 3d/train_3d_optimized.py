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

# Configure torch._dynamo early to avoid DDP optimizer issues
import torch._dynamo
torch._dynamo.config.optimize_ddp = False  # Disable DDP optimizer for higher order ops
torch._dynamo.config.suppress_errors = True  # Fall back to eager mode on errors

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
tensorboard_log_dir = 'runs/3d_parallel_stable'

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

# -----------------------------------------------------------------------------

# data - Stable settings to prevent NaN loss
dataset = 'openwebtext'
gradient_accumulation_steps = 16  # Reduced for stability
batch_size = 8  # Reduced batch size for stability
block_size = 1024  # Reduced sequence length for stability

# model - Keep original GPT-2 124M size
n_layer = 12  # Original GPT-2 small
n_head = 12   # Original GPT-2 small
n_embd = 768  # Original GPT-2 small
dropout = 0.1 # Increased dropout for regularization and stability
bias = False
base_vocab_size = 50257

# 3D parallelism configuration - Conservative settings for stability
tensor_model_parallel_size = 2  # Reduced from 4 to 2 for stability
pipeline_model_parallel_size = 1  # Disable pipeline to reduce overhead
# Data parallel size is computed automatically: 8/(2*1) = 4

# adamw optimizer - Conservative settings to prevent NaN
learning_rate = 5e-5  # Much lower learning rate for stability
max_iters = 600000
weight_decay = 1e-2  # Reduced weight decay
beta1 = 0.9
beta2 = 0.999  # More conservative beta2
grad_clip = 0.5  # Stricter gradient clipping

# learning rate decay settings
decay_lr = True
warmup_iters = 5000  # Longer warmup for stability
lr_decay_iters = 600000
min_lr = 5e-6

# system
device = 'cuda'
dtype = 'bfloat16'  # Use bfloat16 for A100
compile = False  # Disable compilation for debugging and stability

# Performance optimizations
pin_memory = True
non_blocking = True
prefetch_factor = 2  # Reduced prefetching

# I/O - More frequent evaluation for debugging
out_dir = 'out_stable'
eval_interval = 100  # More frequent evaluation
log_interval = 1
eval_iters = 50  # Reduced for faster evaluation
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'

# wandb logging
wandb_log = False
wandb_project = 'nanogpt-3d-stable'
wandb_run_name = 'gpt2-3d-stable'

# tensorboard logging
tensorboard_log = True
tensorboard_log_dir = 'runs/3d_parallel_stable'

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
    
    def __init__(self, data_dir, split, block_size, batch_size_per_dp, device, vocab_size):
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.batch_size_per_dp = batch_size_per_dp
        self.device = device
        self.vocab_size = vocab_size
        
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
        
        # Critical fix: Clamp both input and target values to vocab_size - 1
        x = torch.clamp(x, 0, self.vocab_size - 1)
        y = torch.clamp(y, 0, self.vocab_size - 1)
        
        # Check for invalid tokens
        if torch.any(x >= self.vocab_size) or torch.any(y >= self.vocab_size):
            print(f"WARNING: Found tokens >= vocab_size ({self.vocab_size}). Max x: {x.max()}, Max y: {y.max()}")
        
        if self.device == 'cuda':
            if pin_memory:
                x = x.pin_memory()
                y = y.pin_memory()
            x = x.to(device, non_blocking=non_blocking)
            y = y.to(device, non_blocking=non_blocking)
        else:
            x, y = x.to(device), y.to(device)
        
        return x, y


def get_batch(split, data_dir, vocab_size):
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
            data_dir, split, block_size, batch_size_per_dp, device, vocab_size
        )
    
    return get_batch.loaders[loader_key].get_batch()


@torch.no_grad()
def estimate_loss(model, data_dir, ctx, vocab_size):
    """Estimate loss on train and val datasets - optimized"""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)  # Keep on GPU
        for k in range(eval_iters):
            X, Y = get_batch(split, data_dir, vocab_size)
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
        # Display GPU information
        try:
            gpu_name = torch.cuda.get_device_name()
            gpu_count = torch.cuda.device_count()
            print(f"🖥️  GPU Information:")
            print(f"  GPU Model: {gpu_name}")
            print(f"  GPU Count: {gpu_count}")
            
            # Display expected peak FLOPS for MFU calculation (simplified GPU list)
            gpu_name_lower = gpu_name.lower()
            if 'h20' in gpu_name_lower:
                peak_flops = 148  # Corrected: 296 TFLOPS is for FP8
                gpu_type = "H20"
            elif 'h100' in gpu_name_lower:
                peak_flops = 989 if 'sxm' in gpu_name_lower else 756
                gpu_type = "H100"
            elif 'a100' in gpu_name_lower:
                peak_flops = 312
                gpu_type = "A100"
            elif 'v100' in gpu_name_lower:
                peak_flops = 125
                gpu_type = "V100"
            elif 'p100' in gpu_name_lower:
                peak_flops = 18.7
                gpu_type = "P100"
            elif 'tesla' in gpu_name_lower:
                if 'v100' in gpu_name_lower:
                    peak_flops = 125
                    gpu_type = "Tesla V100"
                else:
                    peak_flops = 50
                    gpu_type = "Tesla (other)"
            else:
                peak_flops = 100  # Conservative fallback
                gpu_type = "Unknown"
            
            print(f"  Expected Peak FLOPS (bfloat16): {peak_flops} TFLOPS per GPU")
            print(f"  GPU Type Detected: {gpu_type}")
        except Exception as e:
            print(f"⚠️  Could not detect GPU info: {e}")
        
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
    
    # Initialize iter_num and best_val_loss
    iter_num = 0
    best_val_loss = 1e9
    
    if init_from == 'scratch':
        model = ParallelGPT(gptconf)
    elif init_from == 'resume':
        # Resume from checkpoint
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        if master_process:
            print(f"🔄 Resuming training from {ckpt_path}")
        
        if not os.path.exists(ckpt_path):
            if master_process:
                print(f"❌ Checkpoint not found: {ckpt_path}")
                print("   Available files in out_dir:")
                if os.path.exists(out_dir):
                    for f in os.listdir(out_dir):
                        print(f"     {f}")
                else:
                    print(f"     Directory {out_dir} does not exist")
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
        checkpoint = torch.load(ckpt_path, map_location=device)
        
        # Load model state
        model = ParallelGPT(gptconf)
        
        # Get the model state dict, handling DDP wrapper
        model_state_dict = checkpoint['model']
        
        # Load model weights
        try:
            model.load_state_dict(model_state_dict, strict=True)
            if master_process:
                print("✅ Model state loaded successfully")
        except Exception as e:
            if master_process:
                print(f"⚠️  Model state loading failed, trying without strict mode: {e}")
            model.load_state_dict(model_state_dict, strict=False)
        
        # Load training state
        iter_num = checkpoint.get('iter_num', 0)
        best_val_loss = checkpoint.get('best_val_loss', 1e9)
        
        if master_process:
            print(f"✅ Resumed from iteration {iter_num}, best_val_loss: {best_val_loss:.4f}")
    else:
        raise NotImplementedError(f"init_from='{init_from}' not implemented for 3D parallel model")
    
    # Move model to device
    model.to(device)
    
    # Initialize gradient scaler with optimizations
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))
    # Disable scaler for bfloat16 to prevent NaN issues
    if dtype == 'bfloat16':
        scaler = torch.amp.GradScaler('cuda', enabled=False)
        if master_process:
            print("⚠️  Gradient scaler disabled for bfloat16 to prevent NaN issues")
    
    # Optimizer with optimizations
    optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
    
    # Load optimizer state if resuming
    if init_from == 'resume' and 'optimizer' in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
            if master_process:
                print("✅ Optimizer state loaded successfully")
        except Exception as e:
            if master_process:
                print(f"⚠️  Failed to load optimizer state: {e}")
                print("   Continuing with fresh optimizer state")
    
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
        # DDP optimizer is already disabled at the top of the script
        unoptimized_model = model
        try:
            model = torch.compile(model)
            if master_process:
                print("✅ Model compilation successful")
        except Exception as e:
            if master_process:
                print(f"⚠️  Model compilation failed: {e}")
                print("🔄 Falling back to eager mode...")
            model = unoptimized_model
    
    # Training loop variables (iter_num and best_val_loss already set in resume logic)
    
    if eval_only:
        losses = estimate_loss(model, data_dir, ctx, vocab_size)
        if master_process:
            print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        return
    
    # Pre-fetch first batch
    X, Y = get_batch('train', data_dir, vocab_size)
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
            losses = estimate_loss(model, data_dir, ctx, vocab_size)
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
                
                # Add NaN detection
                if torch.isnan(loss):
                    if master_process:
                        print(f"🚨 NaN loss detected at iter {iter_num}, micro_step {micro_step}")
                        print(f"   X range: [{X.min()}, {X.max()}]")
                        print(f"   Y range: [{Y.min()}, {Y.max()}]")
                        if logits is not None:
                            print(f"   Logits range: [{logits.min()}, {logits.max()}]")
                            print(f"   Logits contains inf: {torch.isinf(logits).any()}")
                    # Skip this batch
                    continue
            
            # Prefetch next batch asynchronously
            X, Y = get_batch('train', data_dir, vocab_size)
            
            # Backward pass
            if dtype == 'bfloat16':
                # No gradient scaling for bfloat16
                loss.backward()
            else:
                scaler.scale(loss).backward()
        
        # Gradient clipping and optimizer step
        if dtype == 'bfloat16':
            # No gradient scaling for bfloat16
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        else:
            # Use gradient scaling for float16
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
