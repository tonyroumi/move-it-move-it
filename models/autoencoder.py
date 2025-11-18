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
        self.encoder = SkeletalEncoder(adj_init, edge_init, encoder_params=params["encoder"])
        self.decoder = SkeletalDecoder(self.encoder.pooling_hierarchy, 
                                       decoder_params=params["decoder"])
    
    def forward(self, x: torch.Tensor, offset: torch.Tensor):
        latent, _ = self.encoder(x, offset)
        result = self.decoder(latent, offset)
        return latent, result