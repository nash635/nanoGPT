# Resume training with optimized parameters
# Use this to continue training from checkpoint with better settings

# I/O
out_dir = 'out_stable'
init_from = 'resume'  # Resume from checkpoint
eval_interval = 100
log_interval = 10
eval_iters = 50
eval_only = False
always_save_checkpoint = True

# Improved training parameters
learning_rate = 1e-4  # Increase learning rate (was 5e-5)
max_iters = 800000    # Extend training
weight_decay = 1e-2
beta1 = 0.9
beta2 = 0.95  # Better beta2 for stability
grad_clip = 1.0  # Relax gradient clipping

# Learning rate decay settings - extend the schedule
decay_lr = True
warmup_iters = 2000  # Shorter warmup since we're resuming
lr_decay_iters = 800000  # Extend decay schedule
min_lr = 1e-5  # Higher minimum learning rate

# Slightly increase batch processing
gradient_accumulation_steps = 20  # Increase from 16
batch_size = 12  # Increase from 8 if memory allows

# Optional: increase sequence length if memory allows
# block_size = 1536  # Increase from 1024

# wandb logging
wandb_log = False
wandb_project = 'nanogpt-3d-optimized-resume'
wandb_run_name = 'gpt2-3d-optimized-resume'

# tensorboard logging
tensorboard_log = True
tensorboard_log_dir = 'runs/3d_parallel_optimized_resume'
