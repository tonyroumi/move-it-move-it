from .base import SkeletalBase
from .linear import SkeletalLinear

from typing import List, Optional, Literal

import torch
import torch.nn.functional as F

class SkeletalConv(SkeletalBase):
    """
    Skeletal-temporal convolution.

    The weights of this layer are created to learn how each edge aggregates information from its
    neighboring edges over time. 
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
        out_channels_per_joint: int,
        offset_in_channels_per_joint: Optional[int],
        bias: bool,
        kernel_size: int,
        stride: int,
        padding_mode: str,
        dilation: int,
        groups: int,
        type: Literal["static", "dynamic"] = 'static'
    ):
        super().__init__(adj_list, in_channels_per_joint, out_channels_per_joint)
        self.type = type
        self.bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (0, 0)
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)
        self.bias = torch.zeros(self.out_channels)

        self.mask = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)

        if self.type == 'dynamic':
            self.padding = (self.kernel_size - 1) // 2
            self.offset_encoder = SkeletalLinear(self.adj, offset_in_channels_per_joint, out_channels_per_joint)
        
        super()._init_weights()
    
    def forward(self, x: torch.Tensor, offset: Optional[torch.tensor] = None):
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
        
        if offset and self.type == "dynamic":
            offset_out = self.offset_encoder(self.offset)
            offset_out = offset_out.reshape(offset_out.shape + (1, ))
            output += offset_out / 100

        return output
