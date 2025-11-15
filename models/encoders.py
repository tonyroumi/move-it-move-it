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
        last_pool: bool = False
    ):
        super().__init__()
        self.conv = SkeletalConv(adj_list=adj_list, **(conv_params))
        self.act = nn.LeakyReLU(negative_slope=0.2)
        self.pool = SkeletalPooling(edge_list=edge_list, 
                                    channels_per_edge=conv_params["out_channels_per_joint"], 
                                    last_pool=last_pool)

    def forward(self, x: torch.Tensor):
        y = self.conv(x)
        y = self.act(y)
        y, pool_regions, post_adj = self.pool(y)
        return y, pool_regions, post_adj 

class SkeletalEncoder(nn.Module):
    def __init__(
        self,
        adj_init: List[List[int]],
        edge_init: List[Tuple[int]],
        encoder_params: Dict[str, Any],
        type: Literal["static", "dynamic"] = 'static'
    ):
        super().__init__()

        self.adj_init = adj_init
        self.encoder_params = encoder_params
        self.type = type

        # --- Build Block 1 ---
        self.block1 = SkeletalEncBlock(
            adj_list=adj_init,
            edge_list=edge_init,
            conv_params=self.encoder_params[0],
        )

        self.block2: Optional[SkeletalEncBlock] = None

    def forward(self, x: torch.Tensor, offset: Optional[torch.tensor] = None):
        adjs, pool_regions = [], []
        adjs.append(self.adj_init)

        y, pooled_eges, post_adj = self.block1(x, offset[0])

        adjs.append(post_adj)
        pool_regions.append(pooled_eges)

        # --- Lazy init Block 2 ---
        if self.block2 is None and self.type == "dynamic":
            self.block2 = SkeletalEncBlock(
                adj_list=post_adj,
                edge_list=pooled_eges,
                conv_params=self.encoder_params[1],
                last_pool=True).to(y.device)

        # --- Block 2 ---
        y, pooled_eges, post_adj = self.block2(y, offset[1])
        adjs.append(post_adj)
        pool_regions.append(pooled_eges)

        skips = {
            "adjs": adjs,
            "pool_regions": pool_regions,
        }
        return y, skips
    