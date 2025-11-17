from skeletal_ops import SkeletalConv, SkeletalPooling, SkeletalLinear

from typing import Dict, List, Tuple, Optional, Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalEncBlock(nn.Module):
    """
    One block: SkeletalConv -> LeakyReLU -> AvgPooling
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        edge_list: List[Tuple[int]],
        conv_params: Dict[str, Any],
        pool: bool = True,
        last_pool: bool = False,
    ):
        super().__init__()

        self.conv = SkeletalConv(adj_list=adj_list, **(conv_params))
        # The second block of the static encoder does not contain a pooling operation. 
        self.pool = pool if not pool else SkeletalPooling(edge_list=edge_list, 
                                                      channels_per_edge=conv_params["out_channels_per_joint"], 
                                                      last_pool=last_pool)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor]):      
        y = self.conv(x, offset=offset)
        if self.pool:
            y = self.pool(y)
        y = self.act(y)
        return y

class SkeletalEncoder(nn.Module):
    def __init__(
        self,
        adj_init: List[List[int]],
        edge_init: List[Tuple[int]],
        encoder_params: Dict[str, Any],
        type: Literal["static", "dynamic"] = "static"
    ):
        super().__init__()

        self.adj_init = adj_init
        self.edge_init = edge_init
        self.encoder_params = encoder_params
        self.type = type

        self.adjs = [self.adj_init]
        self.edge_lists = [self.edge_init]

        self._init_blocks()
    
    def _init_blocks(self):
        self.block1 = SkeletalEncBlock(
            adj_list=self.adj_init,
            edge_list=self.edge_init,
            **(self.encoder_params["block1"]),
        )

        post_adj = self.block1.pool.new_adj_list
        post_edge_list = self.block1.pool.new_edge_list
        self.adjs.append(post_adj)
        self.edge_lists.append(post_edge_list)

        self.block2 = SkeletalEncBlock(
            adj_list=post_adj,
            edge_list=post_edge_list,
            **(self.encoder_params["block2"]))

    def forward(self, x: torch.Tensor, offset: Optional[List[torch.tensor]] = None):
        intermediate_features = []
        intermediate_features.append(x)

        y = self.block1(x, offset=offset[0] if offset else None)
        intermediate_features.append(y)

        y = self.block2(y, offset=offset[1] if offset else None)
        intermediate_features.append(y)

        return y, intermediate_features
    