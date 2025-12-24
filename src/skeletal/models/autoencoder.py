"""
Skeletal AutoEncoder to learn how to reconstruct original motions and a primal skeleton in latent space.
"""

from .encoder import SkeletalEncoder
from .decoder import SkeletalDecoder

from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalAutoEncoder(nn.Module):
    def __init__(
        self, 
        adj_init: List[List[int]], 
        edge_init: List[List[int]], 
        params: Dict[str, Any]
    ):
        super().__init__()
        self.encoder = SkeletalEncoder(adj_init, edge_init, params=params["encoder"])
        self.decoder = SkeletalDecoder(self.encoder.pooling_hierarchy, 
                                       params=params["decoder"])
    
    def forward(self, x: torch.Tensor, offset: torch.Tensor):
        B, D, T = x.shape #TODO(anthony) fix this. see comments before

        dummy = x.new_zeros(B, 1, T)   # pad the global position row
        x_ext = torch.cat([x, dummy], dim=1)

        latent = self.encoder(x_ext, offset)
        result_ext = self.decoder(latent, offset)

        result = result_ext[:, :D, :]
        return latent, result
