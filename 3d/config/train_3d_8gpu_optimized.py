# Optimized Configuration for 3D parallel training with 8 GPUs (A100)
# Focuses on maximizing GPU utilization and throughput

# 3D parallelism settings - Optimized for A100
tensor_model_parallel_size = 4  # 4-way tensor parallelism (better for large models)
pipeline_model_parallel_size = 1  # Disable pipeline parallelism (reduces bubble overhead)
# Data parallel size will be 2 (8 / (4*1) = 2)

# Data and model settings  
dataset = 'openwebtext'
block_size = 2048  # Increase context length for A100
batch_size = 16    # Increase batch size per GPU for better utilization
gradient_accumulation_steps = 32  # Increase for larger effective batch size

# Model configuration (GPT-2 124M - keep original size)
n_layer = 12      # Original GPT-2 small layers
n_head = 12       # Original GPT-2 small heads  
n_embd = 768      # Original GPT-2 small embedding
dropout = 0.0     # Original dropout setting
bias = False      # Keep bias disabled

# Training settings
learning_rate = 3e-4  # Slightly lower LR for larger model
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 3e-5

# Evaluation and logging
eval_interval = 500   # More frequent evaluation
eval_iters = 100      # Fewer eval iterations to save time
log_interval = 10

# System settings
device = 'cuda'
dtype = 'bfloat16'    # Use bfloat16 for A100 (better than float16)
compile = True        # Enable compilation for better performance

# TensorBoard logging
tensorboard_log = True
tensorboard_log_dir = 'runs/3d_parallel_8gpu_optimized'

# Checkpointing
out_dir = 'out_3d_8gpu_optimized'
always_save_checkpoint = True
init_from = 'scratch'

# Logging
wandb_log = False
wandb_project = 'nanogpt-3d-optimized'
wandb_run_name = 'gpt2-124m-3d-8gpu-optimized'
