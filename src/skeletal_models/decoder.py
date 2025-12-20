from .encoder import PoolingInfo
from src.skeletal_ops import SkeletalConv, SkeletalUnpool

from typing import Dict, List, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalDecBlock(nn.Module):
    """
    One skeletal decoder block : UpS -> SkeletalUnpool -> SkeletalConv -> Optional[LeakyReLU]
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        pooled_edges: List[List[int]],
        channels_per_edge: int,
        conv_params: Dict[str, Any],
        global_pos_inc: bool = True,
        activation: bool = True
    ):
        super().__init__()

        self.UpS = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.unpool = SkeletalUnpool(pooled_edges=pooled_edges, channels_per_edge=channels_per_edge) 
        self.conv = SkeletalConv(adj_list=adj_list, **conv_params, global_pos_inc=global_pos_inc)
        self.act = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):      
        y = self.unpool(self.UpS(x))
        y = self.conv(y, offset)
        y = self.act(y)
        return y

class SkeletalDecoder(nn.Module):
    def __init__(
        self,
        pooled_info: List[PoolingInfo],
        params: Dict[str, Any]
    ):
        super().__init__()

        self.pooled_info = pooled_info
        self.params = params

        self._init_blocks()
    
    def _init_blocks(self):
        self.block1 = SkeletalDecBlock(
            adj_list=self.pooled_info[1].adj_list,
            pooled_edges=self.pooled_info[2].pooled_edges,
            **(self.params["block1"]))

        self.block2 = SkeletalDecBlock(
            adj_list=self.pooled_info[0].adj_list,
            pooled_edges=self.pooled_info[1].pooled_edges,
            **(self.params["block2"]))
        
    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):
        y = self.block1(x, offset=offset[1] if offset else None)
        y = self.block2(y, offset=offset[0] if offset else None)

        return y