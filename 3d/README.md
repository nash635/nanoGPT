# nanoGPT 3D Parallelism

This directory contains an implementation of nanoGPT adapted for Megatron-LM style 3D parallelism, supporting:

- **Tensor Parallelism (TP)**: Splits model weights across multiple GPUs
- **Pipeline Parallelism (PP)**: Splits model layers across multiple GPUs  
- **Data Parallelism (DP)**: Splits training data across multiple GPUs

## Architecture

### Core Components

1. **`megatron/initialize.py`**: 3D parallelism initialization and process group management
2. **`megatron/tensor_parallel.py`**: Tensor parallel layers (attention, MLP, embeddings)
3. **`megatron/pipeline_parallel.py`**: Pipeline parallel utilities and scheduling
4. **`megatron/model.py`**: ParallelGPT model with 3D parallelism support
5. **`train_3d.py`**: Main training script adapted for 3D parallelism

### Directory Structure

```
3d/
├── megatron/
│   ├── __init__.py
│   ├── initialize.py          # 3D parallel initialization
│   ├── tensor_parallel.py     # Tensor parallel layers
│   ├── pipeline_parallel.py   # Pipeline parallel utilities
│   └── model.py               # Parallel GPT model
├── config/
│   ├── train_3d_2gpu.py      # 2 GPU configuration (TP=2)
│   └── train_3d_4gpu.py      # 4 GPU configuration (TP=2, PP=2)
├── train_3d.py               # Training script
├── run_3d_training.sh        # Launch script
└── README.md                 # This file
```

## Usage

### 2 GPU Training (Tensor Parallelism Only)

```bash
# Launch with 2 GPUs using tensor parallelism
./run_3d_training.sh config/train_3d_2gpu.py 2
```

This configuration uses:
- Tensor Parallel Size: 2 (splits model across 2 GPUs)
- Pipeline Parallel Size: 1 (no pipeline parallelism)
- Data Parallel Size: 1 (computed automatically)

### 4 GPU Training (Tensor + Pipeline Parallelism)

```bash
# Launch with 4 GPUs using tensor and pipeline parallelism
./run_3d_training.sh config/train_3d_4gpu.py 4
```

This configuration uses:
- Tensor Parallel Size: 2 (splits model width)
- Pipeline Parallel Size: 2 (splits model depth)
- Data Parallel Size: 1 (computed automatically)

### Manual Launch

You can also launch training manually:

```bash
cd 3d
export CUDA_VISIBLE_DEVICES=0,1
torchrun --standalone --nproc_per_node=2 train_3d.py config/train_3d_2gpu.py
```

## Configuration

### Key Configuration Parameters

- `tensor_model_parallel_size`: Number of GPUs for tensor parallelism
- `pipeline_model_parallel_size`: Number of GPUs for pipeline parallelism
- `batch_size`: Micro-batch size per GPU
- `gradient_accumulation_steps`: Steps to accumulate gradients

### Effective Batch Size Calculation

```
effective_batch_size = batch_size * gradient_accumulation_steps * data_parallel_size
```

Where `data_parallel_size = world_size / (tensor_parallel_size * pipeline_parallel_size)`

## Model Architecture Changes

### Tensor Parallel Layers

1. **VocabParallelEmbedding**: Vocabulary split across GPUs
2. **ColumnParallelLinear**: Column-wise weight splitting (for attention QKV and MLP fc)
3. **RowParallelLinear**: Row-wise weight splitting (for attention output and MLP projection)

### Pipeline Parallel Support

- Model layers are automatically split across pipeline stages
- First stage handles embeddings
- Last stage handles final layer norm and language model head
- Intermediate stages handle transformer blocks

## Memory and Compute Benefits

### Tensor Parallelism
- Reduces memory per GPU: weights split across GPUs
- Maintains compute efficiency: parallel matrix operations
- All-reduce communication for activations

### Pipeline Parallelism  
- Reduces memory per GPU: layers split across GPUs
- Overlaps computation and communication
- Point-to-point communication between stages

### Combined 3D Parallelism
- Maximum memory reduction
- Scales to large numbers of GPUs
- Optimal for very large models

## Performance Considerations

1. **Communication Overhead**: 
   - Tensor parallelism requires all-reduce operations
   - Pipeline parallelism requires point-to-point communication

2. **Load Balancing**:
   - Ensure even layer distribution across pipeline stages
   - Balance tensor parallel splits

3. **Batch Size Tuning**:
   - Smaller micro-batches for pipeline efficiency
   - Larger gradient accumulation to maintain effective batch size

## Limitations

1. **Pipeline Parallelism**: Currently implements basic GPipe scheduling
2. **Checkpoint Loading**: Pretrained model loading not yet implemented
3. **Dynamic Shapes**: Fixed block size and model dimensions
4. **Communication Backend**: Optimized for NCCL, may need adjustments for other backends

## Future Enhancements

1. **Advanced Pipeline Scheduling**: Implement PipeDream-2BW interleaved scheduling
2. **Sequence Parallelism**: Add sequence dimension parallelism for long sequences
3. **Expert Parallelism**: Support for mixture-of-experts models
4. **Gradient Compression**: Implement gradient compression for communication efficiency

## Requirements

- PyTorch with distributed support
- NCCL backend for optimal performance
- Multiple GPUs (2+ recommended)
- Sufficient GPU memory for model partitions

## Troubleshooting

### Common Issues

1. **NCCL Errors**: Set `NCCL_DEBUG=INFO` for detailed logs
2. **Memory Issues**: Reduce batch size or increase parallelism
3. **Hanging**: Check process group initialization and GPU visibility

### Debug Commands

```bash
# Check GPU utilization
nvidia-smi

# Monitor process groups
export NCCL_DEBUG=INFO

# Verify tensor shapes
# Add print statements in model forward pass
```
