# config for training GPT-2 (124M) on 2 GPUs (Tesla P100)
# launch as: torchrun --standalone --nproc_per_node=2 train.py config/train_2gpu.py

# wandb logging (optional, you can disable if not needed)
wandb_log = False  # set to True if you want to use wandb
wandb_project = 'nanoGPT'
wandb_run_name = 'gpt2-124M-2gpu'

# data
dataset = 'openwebtext'

# model settings for GPT-2 124M
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

# training settings adjusted for 2 GPUs with P100 (16GB each)
batch_size = 6  # further reduced to avoid memory issues
block_size = 1024
gradient_accumulation_steps = 5 * 4  # adjusted for 2 GPUs (was 5*8 for 8 GPUs)

# total effective batch size = batch_size * block_size * gradient_accumulation_steps * num_gpus
# = 6 * 1024 * 20 * 2 = 245,760 tokens per step (safer for 2 GPUs)

# learning rate and optimization
learning_rate = 6e-4
max_iters = 600000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# learning rate decay
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

# evaluation
eval_interval = 1000
eval_iters = 200
log_interval = 10

# system
device = 'cuda'
dtype = 'float16'  # use float16 for P100 (doesn't support bfloat16)
compile = False  # disable compilation to avoid memory issues

# DDP settings
backend = 'gloo'  # use gloo instead of nccl for better compatibility

# checkpointing
out_dir = 'out'
always_save_checkpoint = True
init_from = 'scratch'
