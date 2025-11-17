from skeletal_ops import SkeletalConv, SkeletalUnpool

from typing import Dict, List, Tuple, Optional, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalDecBlock(nn.Module):
    """
    One decoder block : UnPool -> UPs -> SkeletalConv -> Optional[LeakyReLU]
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        pooled_edges: List[List[int]],
        channels_per_edge: int,
        conv_params: Dict[str, Any]
    ):
        super().__init__()

        self.unpool = SkeletalUnpool(pooled_regions=pooled_edges, channels_per_edge=channels_per_edge)
        self.UpS = nn.Upsample(scale_factor=2, mode="linear", align_corners=False) 
        self.conv = SkeletalConv(adj_list=adj_list, **conv_params)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):      
        y = self.unpool(self.UpS(x))
        y = self.conv(y, offset)
        y = self.act(y)
        return y

class Decoder(nn.Module):
    def __init__(
        self,
        adj_list: List[List[int]],
        pooled_edges: List[List[int]],
        decoder_params: Dict[str, Any]
    ):
        super().__init__()

        self.adj_lists = adj_list
        self.pooled_edges = pooled_edges
        self.decoder_params = decoder_params

        self.adjs = [self.adj_list]
        self.pooling_list = [self.pooling_list]

        self._init_blocks()
    
    def _init_blocks(self):
        self.block1 = SkeletalDecBlock(
            adj_list=self.adj_lists[0],
            pooled_edges=self.pooled_edges[0],
            **(self.decoder_params["block1"])
        )

        self.block2 = SkeletalDecBlock(
            adj_list=self.adj_lists[1],
            pooled_edges=self.pooled_edges[1],
            **(self.decoder_params["block2"]))
        
    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):
        intermediate_features = []
        intermediate_features.append(x)

        y = self.block1(x, offset=offset[0] if offset else None)
        intermediate_features.append(y)

        y = self.block2(y, offset=offset[1] if offset else None)
        intermediate_features.append(y)

        return y, intermediate_features