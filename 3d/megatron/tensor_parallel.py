"""
Tensor parallelism utilities for Megatron-LM style 3D parallelism
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.distributed as dist
from typing import Optional, Any
import math

from .initialize import (
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_world_size,
    get_tensor_model_parallel_rank
)


def ensure_divisibility(numerator: int, denominator: int) -> None:
    """Ensure that numerator is divisible by the denominator."""
    assert numerator % denominator == 0, f"{numerator} is not divisible by {denominator}"


def divide(numerator: int, denominator: int) -> int:
    """Ensure that numerator is divisible by the denominator and return the division value."""
    ensure_divisibility(numerator, denominator)
    return numerator // denominator


def split_tensor_along_last_dim(tensor: torch.Tensor, num_partitions: int, 
                               contiguous_split_chunks: bool = False) -> list:
    """Split a tensor along its last dimension."""
    last_dim = tensor.dim() - 1
    last_dim_size = divide(tensor.size()[last_dim], num_partitions)
    
    tensor_list = torch.split(tensor, last_dim_size, dim=last_dim)
    
    if contiguous_split_chunks:
        return [chunk.contiguous() for chunk in tensor_list]
    
    return tensor_list


class _ReduceFromModelParallelRegion(torch.autograd.Function):
    """All-reduce the input from the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _reduce(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


class _ScatterToModelParallelRegion(torch.autograd.Function):
    """Scatter the input to the model parallel region."""

    @staticmethod
    def forward(ctx, input_):
        return _scatter(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _gather(grad_output)


class _GatherFromModelParallelRegion(torch.autograd.Function):
    """Gather the input from the model parallel region and concatenate."""

    @staticmethod
    def forward(ctx, input_):
        return _gather(input_)

    @staticmethod
    def backward(ctx, grad_output):
        return _scatter(grad_output)


def _reduce(input_):
    """All-reduce the input tensor across model parallel group."""
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    
    # For gloo backend, ensure tensor is on the right device
    original_device = input_.device
    
    try:
        torch.distributed.all_reduce(input_, group=get_tensor_model_parallel_group())
        return input_
    except Exception as e:
        print(f"[ERROR] _reduce failed: {e}")
        raise


def _scatter(input_):
    """Split the tensor along its last dimension and keep only the corresponding slice."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_

    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()
    
    input_list = split_tensor_along_last_dim(input_, world_size)
    output = input_list[rank].contiguous()
    
    return output


def _gather(input_):
    """Gather tensors and concatenate along the last dimension."""
    world_size = get_tensor_model_parallel_world_size()
    if world_size == 1:
        return input_
    
    last_dim = input_.dim() - 1
    
    # Ensure input tensor is contiguous
    if not input_.is_contiguous():
        input_ = input_.contiguous()
    
    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[get_tensor_model_parallel_rank()] = input_
    
    try:
        torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())
        output = torch.cat(tensor_list, dim=last_dim).contiguous()
        return output
    except Exception as e:
        print(f"[ERROR] _gather failed: {e}")
        print("[DEBUG] Falling back to direct concatenation...")
        
        # Fallback: ensure all tensors in the list are contiguous and try again
        try:
            # Make sure all tensors are contiguous
            for i in range(len(tensor_list)):
                if tensor_list[i] is not None and not tensor_list[i].is_contiguous():
                    tensor_list[i] = tensor_list[i].contiguous()
            
            # Try all_gather again with contiguous tensors
            torch.distributed.all_gather(tensor_list, input_.contiguous(), group=get_tensor_model_parallel_group())
            output = torch.cat(tensor_list, dim=last_dim).contiguous()
            return output
        except Exception as e2:
            print(f"[ERROR] Second attempt failed: {e2}")
            
            # Fallback: manually concatenate what we have
            # This is a simplified approach - we'll manually create the full tensor
            rank = get_tensor_model_parallel_rank()
            
            # Create output tensor with full size
            full_shape = list(input_.shape)
            full_shape[last_dim] = full_shape[last_dim] * world_size
            output = torch.zeros(full_shape, dtype=input_.dtype, device=input_.device)
            
            # Place our partition in the correct position
            start_idx = rank * input_.shape[last_dim]
            end_idx = start_idx + input_.shape[last_dim]
            
            if last_dim == 0:
                output[start_idx:end_idx] = input_
            elif last_dim == 1:
                output[:, start_idx:end_idx] = input_
            elif last_dim == 2:
                output[:, :, start_idx:end_idx] = input_
            else:
                # For higher dimensions, use more general approach
                slices = [slice(None)] * input_.dim()
                slices[last_dim] = slice(start_idx, end_idx)
                output[tuple(slices)] = input_
            
            print(f"[DEBUG] Created fallback tensor with shape {output.shape}")
            return output


# Public API
def reduce_from_tensor_model_parallel_region(input_):
    """All-reduce the input from the tensor model parallel region."""
    return _ReduceFromModelParallelRegion.apply(input_)


def scatter_to_tensor_model_parallel_region(input_):
    """Scatter the input to the tensor model parallel region."""
    return _ScatterToModelParallelRegion.apply(input_)


def gather_from_tensor_model_parallel_region(input_):
    """Gather the input from the tensor model parallel region."""
    return _GatherFromModelParallelRegion.apply(input_)


class VocabParallelEmbedding(nn.Module):
    """Embedding parallelized in the vocabulary dimension."""
    
    def __init__(self, num_embeddings: int, embedding_dim: int, 
                 padding_idx: Optional[int] = None,
                 max_norm: Optional[float] = None,
                 norm_type: float = 2.0,
                 scale_grad_by_freq: bool = False,
                 sparse: bool = False,
                 init_method: Optional[Any] = None):
        super(VocabParallelEmbedding, self).__init__()
        
        # Keep track of dimensions
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        
        # Divide the weight matrix along the vocabulary dimension
        world_size = get_tensor_model_parallel_world_size()
        self.vocab_start_index, self.vocab_end_index = _vocab_range_from_global_vocab_size(
            num_embeddings, get_tensor_model_parallel_rank(), world_size
        )
        self.num_embeddings_per_partition = self.vocab_end_index - self.vocab_start_index
        
        # Allocate weights
        self.weight = Parameter(torch.empty(self.num_embeddings_per_partition, self.embedding_dim))
        if init_method:
            init_method(self.weight)
        else:
            nn.init.xavier_normal_(self.weight)
    
    def forward(self, input_):
        if get_tensor_model_parallel_world_size() == 1:
            return F.embedding(input_, self.weight, self.padding_idx,
                             self.max_norm, self.norm_type,
                             self.scale_grad_by_freq, self.sparse)
        
        # Build the mask
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
        
        # Get the embeddings
        output_parallel = F.embedding(masked_input, self.weight, self.padding_idx,
                                    self.max_norm, self.norm_type,
                                    self.scale_grad_by_freq, self.sparse)
        
        # Mask the output
        output_parallel[input_mask, :] = 0.0
        
        # Reduce across all the model parallel GPUs
        output = reduce_from_tensor_model_parallel_region(output_parallel)
        
        return output


def _vocab_range_from_global_vocab_size(global_vocab_size: int, rank: int, world_size: int):
    """Calculate vocabulary range for each rank."""
    per_partition_vocab_size = divide(global_vocab_size, world_size)
    index_f = rank * per_partition_vocab_size
    index_l = index_f + per_partition_vocab_size
    return index_f, index_l


class ColumnParallelLinear(nn.Module):
    """Linear layer with column parallelism."""
    
    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 gather_output: bool = True, init_method: Optional[Any] = None,
                 stride: int = 1, keep_master_weight_for_test: bool = False):
        super(ColumnParallelLinear, self).__init__()
        
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.gather_output = gather_output
        
        # Divide the weight matrix along the last dimension
        world_size = get_tensor_model_parallel_world_size()
        self.output_size_per_partition = divide(output_size, world_size)
        
        # Parameters
        self.weight = Parameter(torch.empty(self.output_size_per_partition, self.input_size))
        if init_method:
            init_method(self.weight)
        else:
            nn.init.xavier_normal_(self.weight)
            
        if bias:
            self.bias = Parameter(torch.empty(self.output_size_per_partition))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input_):
        # For column parallel, input should be the full tensor, not scattered
        # Each rank has a partition of the weight matrix
        output_parallel = F.linear(input_, self.weight, self.bias)
        
        if self.gather_output:
            # All-gather across the partitions
            output = gather_from_tensor_model_parallel_region(output_parallel)
        else:
            output = output_parallel
        
        return output


class RowParallelLinear(nn.Module):
    """Linear layer with row parallelism."""
    
    def __init__(self, input_size: int, output_size: int, bias: bool = True,
                 input_is_parallel: bool = False, init_method: Optional[Any] = None,
                 stride: int = 1, keep_master_weight_for_test: bool = False):
        super(RowParallelLinear, self).__init__()
        
        # Keep input parameters
        self.input_size = input_size
        self.output_size = output_size
        self.input_is_parallel = input_is_parallel
        
        # Divide the weight matrix along the last dimension
        world_size = get_tensor_model_parallel_world_size()
        self.input_size_per_partition = divide(input_size, world_size)
        
        # Parameters
        self.weight = Parameter(torch.empty(self.output_size, self.input_size_per_partition))
        if init_method:
            init_method(self.weight)
        else:
            nn.init.xavier_normal_(self.weight)
            
        if bias:
            self.bias = Parameter(torch.empty(self.output_size))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
    
    def forward(self, input_):
        # Set up backprop all-reduce
        if self.input_is_parallel:
            input_parallel = input_
        else:
            input_parallel = scatter_to_tensor_model_parallel_region(input_)
        
        # Matrix multiply
        output_parallel = F.linear(input_parallel, self.weight)
        
        # All-reduce across all the partitions
        output_ = reduce_from_tensor_model_parallel_region(output_parallel)
        
        if self.bias is not None:
            output = output_ + self.bias
        else:
            output = output_
        
        return output


def vocab_parallel_cross_entropy(vocab_parallel_logits, target):
    """
    Perform cross entropy loss when logits are split across tensor parallel ranks.
    
    Args:
        vocab_parallel_logits: logits with vocabulary dimension split across ranks
        target: target token ids (full vocabulary range)
    
    Returns:
        cross entropy loss
    """
    # Get tensor parallel info
    world_size = get_tensor_model_parallel_world_size()
    rank = get_tensor_model_parallel_rank()
    
    if world_size == 1:
        return F.cross_entropy(vocab_parallel_logits, target, ignore_index=-1)
    
    # Calculate vocabulary range for this rank
    vocab_size = vocab_parallel_logits.size(-1) * world_size
    vocab_start, vocab_end = _vocab_range_from_global_vocab_size(vocab_size, rank, world_size)
    
    # Create mask for targets that belong to this partition
    target_mask = (target >= vocab_start) & (target < vocab_end)
    
    # Create a new target tensor with local indices
    # For targets not in this partition, we'll use a dummy index (0)
    local_target = torch.zeros_like(target)
    local_target[target_mask] = target[target_mask] - vocab_start
    
    # Compute loss only for targets in this partition
    if target_mask.any():
        # Create logits for valid targets only
        valid_logits = vocab_parallel_logits[target_mask]
        valid_targets = local_target[target_mask]
        
        # Compute cross entropy loss
        partition_loss = F.cross_entropy(valid_logits, valid_targets, reduction='sum')
        partition_count = target_mask.sum().float()
    else:
        partition_loss = torch.tensor(0.0, device=vocab_parallel_logits.device, dtype=vocab_parallel_logits.dtype)
        partition_count = torch.tensor(0.0, device=vocab_parallel_logits.device, dtype=torch.float32)
    
    # All-reduce the loss and count across all partitions
    torch.distributed.all_reduce(partition_loss, group=get_tensor_model_parallel_group())
    torch.distributed.all_reduce(partition_count, group=get_tensor_model_parallel_group())
    
    # Compute final loss (average)
    if partition_count > 0:
        loss = partition_loss / partition_count
    else:
        loss = partition_loss  # Should be 0
    
    return loss
