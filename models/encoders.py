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
        pool_regions, post_edge_list, post_adj = None, None, None
        
        y = self.conv(x, offset=offset)
        if self.pool:
            y, pool_regions, post_edge_list, post_adj = self.pool(y)
        y = self.act(y)
        return y, pool_regions, post_edge_list, post_adj 

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

        # --- Build Block 1 ---
        self.block1 = SkeletalEncBlock(
            adj_list=adj_init,
            edge_list=edge_init,
            **(self.encoder_params["block1"]),
        )

        self.block2: Optional[SkeletalEncBlock] = None

    def forward(self, x: torch.Tensor, offset: Optional[List[torch.tensor]] = None):
        intermediate_features, adjs, edges, pool_regions = [], [], [], []
        intermediate_features.append(x)
        adjs.append(self.adj_init)
        edges.append(self.edge_init)

        y, pooled_edges, post_edge_list, post_adj = self.block1(x, offset=offset[0] if offset else None)
        intermediate_features.append(y)
        adjs.append(post_adj)
        edges.append(post_edge_list)
        pool_regions.append(pooled_edges)

        # --- Lazy init Block 2 ---
        if self.block2 is None:
            self.block2 = SkeletalEncBlock(
                adj_list=post_adj,
                edge_list=post_edge_list,
                **(self.encoder_params["block2"])).to(y.device)

        # --- Block 2 ---
        y, pooled_edges, post_edge_list, post_adj = self.block2(y, offset=offset[1] if offset else None)
        intermediate_features.append(y)
        adjs.append(post_adj)
        edges.append(post_edge_list)
        pool_regions.append(pooled_edges)

        skips = {
            "intermediate_features": intermediate_features,
            "adjs": adjs,
            "pool_regions": pool_regions,
        }

        return y, skips
    