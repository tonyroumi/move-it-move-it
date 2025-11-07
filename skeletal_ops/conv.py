from .base import SkeletalBase
from .linear import SkeletonLinear

from typing import List, Optional

import torch
import torch.nn.functional as F

class SkeletalConv(SkeletalBase):
    """
    Skeletal-temporal convolution.
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
        out_channels_per_joint: int,
        bias: bool,
        kernel_size: int,
        stride: int,
        padding: int,
        padding_mode: str,
        dilation: int,
        groups: int
    ):
        super().__init__(adj_list, in_channels_per_joint, out_channels_per_joint)

        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding)
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)
        self.bias = torch.zeros(self.out_channels)

        self.mask = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)

        offset_in_channels = 0 * self.E
        self.offset_encoder = SkeletonLinear(self.adj, offset_in_channels, out_channels_per_joint)
        
        super()._init_weights()
    
    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):
        """
        Args:
            x:      [B, C, T]
            offset: [B, C]
        """
        weight_masked = self.weight * self.mask

        x = F.pad(x, pad=self.padding, mode=self.padding_mode)
        output = F.conv1d(x, 
                          weight_masked, 
                          self.bias, 
                          self.stride,
                          self.dilation, 
                          self.groups) 
        
        if offset:
            offset_out = self.offset_encoder(self.offset)
            offset_out = offset_out.reshape(offset_out.shape + (1, ))
            output += offset_out / 100

        return output
