# Configuration for 3D parallel training with 8 GPUs (single node)
# Tensor Parallel (TP) + Data Parallel (DP) configuration

# 3D parallelism settings
tensor_model_parallel_size = 2  # 2-way tensor parallelism
pipeline_model_parallel_size = 2  # 2-way pipeline parallelism
# Data parallel size will be 2 (8 / (2*2) = 2)

# Data and model settings  
dataset = 'openwebtext'
batch_size = 8  # Larger batch size for more GPUs
block_size = 1024
gradient_accumulation_steps = 20  # Adjusted for memory and throughput

# Model configuration (GPT-2 124M)
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# Training settings
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# Learning rate schedule
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# Evaluation and logging
eval_interval = 1000
eval_iters = 200
log_interval = 10

# System settings
device = 'cuda'
dtype = 'float16'  # P100 doesn't support bfloat16
compile = False  # Disabled for compatibility

# TensorBoard logging
tensorboard_log = True
tensorboard_log_dir = 'runs/3d_parallel_8gpu'

# Checkpointing
out_dir = 'out_3d_8gpu'
always_save_checkpoint = True
init_from = 'scratch'

# Logging
wandb_log = False
wandb_project = 'nanogpt-3d'
wandb_run_name = 'gpt2-124m-3d-8gpu'
