from typing import Dict, List, Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalBase(nn.Module):
    """
    Base class for some skeletal ops.
    """
    def __init__(
        self,
        adj_list: Dict[int, List[int]],
        in_channels_per_joint: int,
        out_channels_per_joint: int
    ):
       super().__init__()
       self.adj = adj_list
       self.edges = sorted(self.adj.keys())
       self.E = len(self.edges)

       self.in_channels = in_channels_per_joint * self.E
       self.out_channels = out_channels_per_joint * self.E 

       # Map each neighbor edge to its channel range. 
       # E.x. (in_channels_per_joint = 2)
       # adj = {0: [0,1], 1: [0]} 
       # expanded_adj_list = {0: [0,1,2,3], 1: [0,1]}

       self.expanded_adj_list = [
           [k * in_channels_per_joint + i 
            for k in neighbor 
            for i in range(in_channels_per_joint)]
            for neighbor in self.adj
       ]

    def _init_weights(self):
        #go ahead and take the time to understand how they initialize parameters.
        pass




