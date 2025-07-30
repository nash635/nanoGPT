"""
Megatron-LM 3D Parallelism Initialization
包含数据并行(DP)、张量并行(TP)和流水线并行(PP)的初始化
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple
import os


class ParallelState:
    """全局并行状态管理"""
    
    # Process groups
    _MODEL_PARALLEL_GROUP = None
    _TENSOR_MODEL_PARALLEL_GROUP = None
    _PIPELINE_MODEL_PARALLEL_GROUP = None
    _DATA_PARALLEL_GROUP = None
    
    # World size and ranks
    _TENSOR_MODEL_PARALLEL_WORLD_SIZE = None
    _PIPELINE_MODEL_PARALLEL_WORLD_SIZE = None
    _DATA_PARALLEL_WORLD_SIZE = None
    
    _TENSOR_MODEL_PARALLEL_RANK = None
    _PIPELINE_MODEL_PARALLEL_RANK = None
    _DATA_PARALLEL_RANK = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    virtual_pipeline_model_parallel_size: Optional[int] = None,
    pipeline_model_parallel_split_rank: Optional[int] = None,
) -> None:
    """
    Initialize 3D model parallelism.
    
    Args:
        tensor_model_parallel_size: Tensor model parallel size
        pipeline_model_parallel_size: Pipeline model parallel size
        virtual_pipeline_model_parallel_size: Virtual pipeline parallel size
        pipeline_model_parallel_split_rank: Pipeline split rank
    """
    
    # Get world size and rank
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    # Calculate data parallel size
    data_parallel_size = world_size // (tensor_model_parallel_size * pipeline_model_parallel_size)
    
    print(f"3D Parallelism Configuration:")
    print(f"  World size: {world_size}")
    print(f"  Tensor parallel size: {tensor_model_parallel_size}")
    print(f"  Pipeline parallel size: {pipeline_model_parallel_size}")
    print(f"  Data parallel size: {data_parallel_size}")
    
    # Validate configuration
    expected_world_size = tensor_model_parallel_size * pipeline_model_parallel_size * data_parallel_size
    assert world_size == expected_world_size, \
        f"World size {world_size} != expected {expected_world_size}"
    
    # Build process groups
    _build_model_parallel_groups(
        tensor_model_parallel_size,
        pipeline_model_parallel_size,
        data_parallel_size
    )


def _build_model_parallel_groups(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    data_parallel_size: int
) -> None:
    """Build the model parallel process groups."""
    
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    
    print(f"Rank {rank}: Building process groups...")
    
    # Build tensor model parallel groups
    num_tensor_model_parallel_groups = world_size // tensor_model_parallel_size
    for i in range(num_tensor_model_parallel_groups):
        start_rank = i * tensor_model_parallel_size
        end_rank = (i + 1) * tensor_model_parallel_size
        ranks = list(range(start_rank, end_rank))
        print(f"Rank {rank}: Creating tensor parallel group with ranks {ranks}")
        group = dist.new_group(ranks)
        if rank in ranks:
            ParallelState._TENSOR_MODEL_PARALLEL_GROUP = group
            print(f"Rank {rank}: Assigned to tensor parallel group")
    
    # Synchronization barrier
    dist.barrier()
    print(f"Rank {rank}: Tensor parallel groups created")
    
    # Build pipeline model parallel groups
    for i in range(pipeline_model_parallel_size):
        ranks = []
        for j in range(data_parallel_size):
            for k in range(tensor_model_parallel_size):
                rank_id = (j * pipeline_model_parallel_size * tensor_model_parallel_size + 
                          i * tensor_model_parallel_size + k)
                ranks.append(rank_id)
        print(f"Rank {rank}: Creating pipeline parallel group with ranks {ranks}")
        group = dist.new_group(ranks)
        if rank in ranks:
            ParallelState._PIPELINE_MODEL_PARALLEL_GROUP = group
            print(f"Rank {rank}: Assigned to pipeline parallel group")
    
    # Synchronization barrier  
    dist.barrier()
    print(f"Rank {rank}: Pipeline parallel groups created")
    
    # Build data parallel groups
    for i in range(data_parallel_size):
        for j in range(tensor_model_parallel_size):
            ranks = []
            for k in range(pipeline_model_parallel_size):
                rank_id = (i * pipeline_model_parallel_size * tensor_model_parallel_size + 
                          k * tensor_model_parallel_size + j)
                ranks.append(rank_id)
            print(f"Rank {rank}: Creating data parallel group with ranks {ranks}")
            group = dist.new_group(ranks)
            if rank in ranks:
                ParallelState._DATA_PARALLEL_GROUP = group
                print(f"Rank {rank}: Assigned to data parallel group")
    
    # Final synchronization barrier
    dist.barrier()
    print(f"Rank {rank}: All process groups created")
    
    # Set world sizes and ranks
    ParallelState._TENSOR_MODEL_PARALLEL_WORLD_SIZE = tensor_model_parallel_size
    ParallelState._PIPELINE_MODEL_PARALLEL_WORLD_SIZE = pipeline_model_parallel_size
    ParallelState._DATA_PARALLEL_WORLD_SIZE = data_parallel_size
    
    ParallelState._TENSOR_MODEL_PARALLEL_RANK = dist.get_rank(ParallelState._TENSOR_MODEL_PARALLEL_GROUP)
    ParallelState._PIPELINE_MODEL_PARALLEL_RANK = dist.get_rank(ParallelState._PIPELINE_MODEL_PARALLEL_GROUP)
    ParallelState._DATA_PARALLEL_RANK = dist.get_rank(ParallelState._DATA_PARALLEL_GROUP)
    
    print(f"Rank {rank}: All ranks assigned")


def destroy_model_parallel():
    """Destroy model parallel groups."""
    global _MODEL_PARALLEL_GROUP
    global _TENSOR_MODEL_PARALLEL_GROUP
    global _PIPELINE_MODEL_PARALLEL_GROUP
    global _DATA_PARALLEL_GROUP
    
    ParallelState._MODEL_PARALLEL_GROUP = None
    ParallelState._TENSOR_MODEL_PARALLEL_GROUP = None
    ParallelState._PIPELINE_MODEL_PARALLEL_GROUP = None
    ParallelState._DATA_PARALLEL_GROUP = None


# Getter functions
def get_tensor_model_parallel_group():
    """Get tensor model parallel group."""
    return ParallelState._TENSOR_MODEL_PARALLEL_GROUP


def get_pipeline_model_parallel_group():
    """Get pipeline model parallel group."""
    return ParallelState._PIPELINE_MODEL_PARALLEL_GROUP


def get_data_parallel_group():
    """Get data parallel group."""
    return ParallelState._DATA_PARALLEL_GROUP


def get_tensor_model_parallel_world_size():
    """Get tensor model parallel world size."""
    return ParallelState._TENSOR_MODEL_PARALLEL_WORLD_SIZE


def get_pipeline_model_parallel_world_size():
    """Get pipeline model parallel world size."""
    return ParallelState._PIPELINE_MODEL_PARALLEL_WORLD_SIZE


def get_data_parallel_world_size():
    """Get data parallel world size."""
    return ParallelState._DATA_PARALLEL_WORLD_SIZE


def get_tensor_model_parallel_rank():
    """Get tensor model parallel rank."""
    return ParallelState._TENSOR_MODEL_PARALLEL_RANK


def get_pipeline_model_parallel_rank():
    """Get pipeline model parallel rank."""
    return ParallelState._PIPELINE_MODEL_PARALLEL_RANK


def get_data_parallel_rank():
    """Get data parallel rank."""
    return ParallelState._DATA_PARALLEL_RANK


def is_pipeline_first_stage():
    """Check if current rank is first pipeline stage."""
    return get_pipeline_model_parallel_rank() == 0


def is_pipeline_last_stage():
    """Check if current rank is last pipeline stage."""
    return get_pipeline_model_parallel_rank() == (get_pipeline_model_parallel_world_size() - 1)
