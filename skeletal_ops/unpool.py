from typing import Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class SkeletalUnpooling(nn.Module):
    """
    Skeletal unpooling layer.

    Args
    ----
    prev_adj        :   Dict[int, List[int]]
                        Adjacency map before pooling (high-resolution skeleton).

    pool_regions    :   List[List[int]]
                        Pooling regions obtained from SkeletalPooling.

    upsampling_params : Dict[str, Tuple[int]]
                        Temporal upsampling parameters for dynamic branch
                        (e.g., {'scale_factor': (1, 2), 'mode': 'nearest'}).

    Notes
    -----
    - "UP"  : Structural unpooling (copy features back to each node).
    - "UpS" : Temporal upsampling applied on dynamic branch only.
    - No learnable parameters. Used to restore feature resolution.
    """

    def __init__(
        self,
        prev_adj: Dict[int, List[int]],
        pool_regions: List[List[int]],
        upsampling_params: Dict[str, Tuple[int]] = None
    ):
        super().__init__()
        self.prev_adj = prev_adj
        self.pool_regions = pool_regions
        self.upsampling_params = upsampling_params

        # Build mapping from region → nodes and node → region
        self.region_to_nodes = {i: region for i, region in enumerate(pool_regions)}
        self.node_to_region = {n: i for i, region in enumerate(pool_regions) for n in region}

    def _structural_unpool(self, x: torch.Tensor) -> torch.Tensor:
        """
        Structural unpooling (UP): copies pooled features back to their original nodes.

        Static:  [B, R, C] -> [B, N, C]
        Dynamic: [B, T, R, C] -> [B, T, N, C]
        """
        if x.dim() == 3:  # static
            B, R, C = x.shape
            N = len(self.node_to_region)
            out = torch.zeros((B, N, C), device=x.device, dtype=x.dtype)
            for node, region_idx in self.node_to_region.items():
                out[:, node, :] = x[:, region_idx, :]
            return out

        elif x.dim() == 4:  # dynamic
            B, T, R, C = x.shape
            N = len(self.node_to_region)
            out = torch.zeros((B, T, N, C), device=x.device, dtype=x.dtype)
            for node, region_idx in self.node_to_region.items():
                out[:, :, node, :] = x[:, :, region_idx, :]
            return out

    def _temporal_upsample(self, x: torch.Tensor) -> torch.Tensor:
        """
        Temporal upsampling (UpS) for dynamic branch only.
        """
        if self.upsampling_params is None:
            return x
        # Expect [B, T, N, C] → interpolate along temporal dimension (dim=1)
        x = F.interpolate(
            x.permute(0, 3, 2, 1),  # [B, C, N, T]
            **self.upsampling_params
        ).permute(0, 3, 2, 1)       # back to [B, T, N, C]
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_up = self._structural_unpool(x)
        if x.dim() == 4:  # dynamic
            x_up = self._temporal_upsample(x_up)
        return x_up
