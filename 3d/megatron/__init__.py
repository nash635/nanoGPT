"""
Megatron-LM style 3D parallelism package for nanoGPT
"""

from .initialize import initialize_model_parallel, destroy_model_parallel
from .tensor_parallel import (
    VocabParallelEmbedding,
    ColumnParallelLinear, 
    RowParallelLinear
)
from .pipeline_parallel import PipelineStage, PipelineEngine
from .model import ParallelGPT, GPTConfig

__all__ = [
    'initialize_model_parallel',
    'destroy_model_parallel', 
    'VocabParallelEmbedding',
    'ColumnParallelLinear',
    'RowParallelLinear',
    'PipelineStage',
    'PipelineEngine', 
    'ParallelGPT',
    'GPTConfig'
]
