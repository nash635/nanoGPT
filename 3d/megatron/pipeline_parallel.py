"""
Pipeline parallelism utilities for Megatron-LM style 3D parallelism
"""

import torch
import torch.nn as nn
from typing import List, Optional, Callable, Any, Union
import torch.distributed as dist
from collections import OrderedDict

from .initialize import (
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_world_size,
    get_pipeline_model_parallel_rank,
    is_pipeline_first_stage,
    is_pipeline_last_stage
)


class PipelineStage(nn.Module):
    """Base class for pipeline stages."""
    
    def __init__(self, layers: nn.ModuleList, stage_id: int):
        super().__init__()
        self.layers = layers
        self.stage_id = stage_id
        
    def forward(self, hidden_states, attention_mask=None):
        for layer in self.layers:
            if hasattr(layer, 'forward'):
                if attention_mask is not None:
                    hidden_states = layer(hidden_states, attention_mask)
                else:
                    hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(hidden_states)
        return hidden_states


def split_model_into_stages(model: nn.Module, num_stages: int) -> List[PipelineStage]:
    """Split a model into pipeline stages."""
    
    # Get all transformer layers
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        layers = model.transformer.h
    elif hasattr(model, 'layers'):
        layers = model.layers
    else:
        raise ValueError("Could not find transformer layers in model")
    
    num_layers = len(layers)
    layers_per_stage = num_layers // num_stages
    remainder = num_layers % num_stages
    
    stages = []
    start_idx = 0
    
    for stage_id in range(num_stages):
        # Calculate number of layers for this stage
        num_layers_this_stage = layers_per_stage
        if stage_id < remainder:
            num_layers_this_stage += 1
        
        # Extract layers for this stage
        end_idx = start_idx + num_layers_this_stage
        stage_layers = nn.ModuleList(layers[start_idx:end_idx])
        
        # Create pipeline stage
        stage = PipelineStage(stage_layers, stage_id)
        stages.append(stage)
        
        start_idx = end_idx
    
    return stages


class PipelineEngine:
    """Pipeline execution engine for forward and backward passes."""
    
    def __init__(self, stages: List[PipelineStage], micro_batch_size: int):
        self.stages = stages
        self.micro_batch_size = micro_batch_size
        self.current_stage = get_pipeline_model_parallel_rank()
        self.num_stages = get_pipeline_model_parallel_world_size()
        
        # Communication buffers
        self.input_buffers = []
        self.output_buffers = []
        
    def forward_step(self, input_tensor: torch.Tensor, 
                    attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Execute forward step for current pipeline stage."""
        
        if self.current_stage < len(self.stages):
            stage = self.stages[self.current_stage]
            output = stage(input_tensor, attention_mask)
            return output
        else:
            return input_tensor
    
    def send_forward(self, output_tensor: torch.Tensor) -> None:
        """Send output tensor to next pipeline stage."""
        if not is_pipeline_last_stage():
            next_rank = self.current_stage + 1
            dist.send(output_tensor, dst=next_rank, group=get_pipeline_model_parallel_group())
    
    def recv_forward(self, tensor_shape: List[int], dtype: torch.dtype = torch.float) -> torch.Tensor:
        """Receive input tensor from previous pipeline stage."""
        if not is_pipeline_first_stage():
            prev_rank = self.current_stage - 1
            tensor = torch.empty(tensor_shape, dtype=dtype, device=torch.cuda.current_device())
            dist.recv(tensor, src=prev_rank, group=get_pipeline_model_parallel_group())
            return tensor
        else:
            return None
    
    def send_backward(self, input_grad: torch.Tensor) -> None:
        """Send gradient tensor to previous pipeline stage."""
        if not is_pipeline_first_stage():
            prev_rank = self.current_stage - 1
            dist.send(input_grad, dst=prev_rank, group=get_pipeline_model_parallel_group())
    
    def recv_backward(self, tensor_shape: List[int], dtype: torch.dtype = torch.float) -> torch.Tensor:
        """Receive gradient tensor from next pipeline stage."""
        if not is_pipeline_last_stage():
            next_rank = self.current_stage + 1
            grad_tensor = torch.empty(tensor_shape, dtype=dtype, device=torch.cuda.current_device())
            dist.recv(grad_tensor, src=next_rank, group=get_pipeline_model_parallel_group())
            return grad_tensor
        else:
            return None


class PipelineSchedule:
    """Pipeline scheduling strategies."""
    
    @staticmethod
    def gpipe_schedule(engine: PipelineEngine, input_data: torch.Tensor, 
                      target: Optional[torch.Tensor] = None, 
                      num_microbatches: int = 4) -> torch.Tensor:
        """GPipe scheduling strategy."""
        
        batch_size = input_data.shape[0]
        micro_batch_size = batch_size // num_microbatches
        
        outputs = []
        
        # Forward pass
        for i in range(num_microbatches):
            start_idx = i * micro_batch_size
            end_idx = (i + 1) * micro_batch_size
            micro_batch = input_data[start_idx:end_idx]
            
            # Receive from previous stage
            if not is_pipeline_first_stage():
                micro_batch = engine.recv_forward(micro_batch.shape, micro_batch.dtype)
            
            # Forward step
            output = engine.forward_step(micro_batch)
            
            # Send to next stage
            if not is_pipeline_last_stage():
                engine.send_forward(output)
            else:
                outputs.append(output)
        
        if is_pipeline_last_stage():
            return torch.cat(outputs, dim=0)
        else:
            return None
    
    @staticmethod
    def interleaved_schedule(engine: PipelineEngine, input_data: torch.Tensor,
                           num_microbatches: int = 4, num_model_chunks: int = 2) -> torch.Tensor:
        """Interleaved pipeline scheduling (PipeDream-2BW style)."""
        
        batch_size = input_data.shape[0]
        micro_batch_size = batch_size // num_microbatches
        
        # This is a simplified version - full implementation would be more complex
        return PipelineSchedule.gpipe_schedule(engine, input_data, num_microbatches=num_microbatches)


def get_pipeline_model_parallel_first_rank():
    """Get the first rank in pipeline parallel group."""
    return get_pipeline_model_parallel_rank() - get_pipeline_model_parallel_rank()


def get_pipeline_model_parallel_last_rank():
    """Get the last rank in pipeline parallel group."""
    last_rank = get_pipeline_model_parallel_world_size() - 1
    return last_rank


def get_pipeline_model_parallel_next_rank():
    """Get the next rank in pipeline parallel group."""
    rank = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return (rank + 1) % world_size


def get_pipeline_model_parallel_prev_rank():
    """Get the previous rank in pipeline parallel group."""
    rank = get_pipeline_model_parallel_rank()
    world_size = get_pipeline_model_parallel_world_size()
    return (rank - 1) % world_size


class PipelineParallelGPT(nn.Module):
    """GPT model with pipeline parallelism."""
    
    def __init__(self, gpt_model, num_pipeline_stages: int):
        super().__init__()
        self.num_pipeline_stages = num_pipeline_stages
        self.pipeline_rank = get_pipeline_model_parallel_rank()
        
        # Split model into stages
        self.stages = split_model_into_stages(gpt_model, num_pipeline_stages)
        
        # Only keep the stage for current rank
        if self.pipeline_rank < len(self.stages):
            self.current_stage = self.stages[self.pipeline_rank]
        else:
            self.current_stage = None
        
        # Store original model components for first and last stages
        if is_pipeline_first_stage():
            self.wte = gpt_model.transformer.wte  # Token embeddings
            self.wpe = gpt_model.transformer.wpe  # Position embeddings
            self.drop = gpt_model.transformer.drop  # Dropout
        
        if is_pipeline_last_stage():
            self.ln_f = gpt_model.transformer.ln_f  # Final layer norm
            self.lm_head = gpt_model.lm_head  # Language model head
    
    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        
        if is_pipeline_first_stage():
            # Token and position embeddings
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            tok_emb = self.wte(idx)  # token embeddings of shape (b, t, n_embd)
            pos_emb = self.wpe(pos)  # position embeddings of shape (t, n_embd)
            x = self.drop(tok_emb + pos_emb)
        else:
            # Receive from previous stage
            x = self.recv_from_prev_stage()
        
        # Forward through current stage
        if self.current_stage is not None:
            x = self.current_stage(x)
        
        if is_pipeline_last_stage():
            # Final layer norm and language model head
            x = self.ln_f(x)
            
            if targets is not None:
                # Training mode - compute loss
                logits = self.lm_head(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
                return loss
            else:
                # Inference mode
                logits = self.lm_head(x[:, [-1], :])  # Only last token
                return logits
        else:
            # Send to next stage
            self.send_to_next_stage(x)
            return None
    
    def recv_from_prev_stage(self):
        """Receive tensor from previous pipeline stage."""
        # This would be implemented with proper tensor shapes and communication
        pass
    
    def send_to_next_stage(self, tensor):
        """Send tensor to next pipeline stage."""
        # This would be implemented with proper communication
        pass
