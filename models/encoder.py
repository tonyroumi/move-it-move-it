from skeletal_ops import SkeletalConv, SkeletalPooling, SkeletalLinear

from dataclasses import dataclass, astuple
from typing import Dict, List, Tuple, Optional, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PoolingInfo:
    """Contains all information needed for pooling/unpooling operations"""
    adj_list: List[List[int]]
    edge_list: List[List[int]]
    pooled_edges: List[List[int]] = None
    
    def to_dict(self):
        return {
            'adj_list': self.adj_list,
            'edge_list': self.edge_list,
            'pooled_edges': self.pooled_edges
        }
    
    def __iter__(self):
        return iter(astuple(self))


class SkeletalEncBlock(nn.Module):
    """
    One block: SkeletalConv -> LeakyReLU -> Optional[AvgPooling]
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        edge_list: List[Tuple[int]],
        conv_params: Dict[str, Any],
        pool_features: bool = True,
        last_pool: bool = False,
    ):
        super().__init__()

        self.conv = SkeletalConv(adj_list=adj_list, **(conv_params))
        # The second block of the static encoder does not contain a pooling operation. 
        self.pool = SkeletalPooling(edge_list=edge_list, 
                                    channels_per_edge=conv_params["out_channels_per_joint"], 
                                    last_pool=last_pool)
        self.act = nn.LeakyReLU(negative_slope=0.2)

        self.pooling_info = PoolingInfo(
            adj_list=self.pool.new_adj_list,
            edge_list= self.pool.new_edge_list,
            pooled_edges=self.pool.pooled_edges
        )

        self.pool_features = pool_features

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):      
        y = self.conv(x, offset=offset)
        if self.pool_features:
            y = self.pool(y)
        y = self.act(y)
        return y


class SkeletalEncoder(nn.Module):
    def __init__(
        self,
        adj_init: List[List[int]],
        edge_init: List[Tuple[int]],
        encoder_params: Dict[str, Any]
    ):
        super().__init__()

        self.adj_init = adj_init
        self.edge_init = edge_init
        self.encoder_params = encoder_params

        self.pooling_hierarchy = [PoolingInfo(adj_list=adj_init, edge_list=edge_init)]

        self._init_blocks()
    
    def _init_blocks(self):
        self.block1 = SkeletalEncBlock(
            adj_list=self.adj_init,
            edge_list=self.edge_init,
            **(self.encoder_params["block1"])
        )
        post_adj, post_edge_list,_ = self.block1.pooling_info

        self.block2 = SkeletalEncBlock(
            adj_list=post_adj,
            edge_list=post_edge_list,
            **(self.encoder_params["block2"])
        )

        self.pooling_hierarchy.append(self.block1.pooling_info)
        self.pooling_hierarchy.append(self.block2.pooling_info)

    def forward(self, x: torch.Tensor, offset: Optional[List[torch.tensor]] = None):
        intermediate_features = [x]

        y = self.block1(x, offset=offset[0] if offset else None)
        intermediate_features.append(y)

        y = self.block2(y, offset=offset[1] if offset else None)
        intermediate_features.append(y)

        return y, intermediate_features
    