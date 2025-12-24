from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn

from transformers.activations import ACT2FN

from src.shared_core import SharedCoreLayer


class SharedCoreExperts(nn.Module):
    def __init__(self, config, intermediate_size=None):
        
        self.config = config
        self.compression_config = config.compression_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size if intermediate_size is not None else config.intermediate_size
        self.num_experts = config.num_experts
        
        self.gate_shared_core = SharedCoreLayer(
            num_experts=self.num_experts,
            d_in=self.hidden_size,
            d_out=self.intermediate_size,
            rank=self.compression_config.rank,
            bias=False)
        self.up_shared_core = SharedCoreLayer(
            num_experts=self.num_experts,
            d_in=self.intermediate_size,
            d_out=self.hidden_size,
            rank=self.compression_config.rank,
            bias=False)
        self.down_shared_core = SharedCoreLayer(
            num_experts=self.num_experts,
            d_in=self.hidden_size,
            d_out=self.intermediate_size,
            rank=self.compression_config.rank,
            bias=True)
        
        self.activation_fn = ACT2FN[config.activation_function]
    
    def forward(self, x, expert_idx):
        
        down_proj = self.down_shared_core(
            self.activation_fn(self.gate_shared_core(x, expert_idx)) * self.up_shared_core(x, expert_idx),
            expert_idx
        )
        
        return down_proj
            
            