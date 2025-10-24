import torch
import torch.nn as nn
from typing import Dict, List

#Each armature has it's own conv layer and bias.
#This is the only thing that is shared. 

class SkeletalConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        num_armatures: int,
        kernel_size: int,
        learned_filters: int, 
        mapping: Dict[int, List[int]],
        activation_fn: any
    ):
        super().__init__()
        self.adjacency = mapping
        
        self._init_net()

    def _init_net(self):
        layers = []
        for i in range(num_armatures):
            layers.append(
                nn.Conv1d(
                    in_channels=in_channels if i == 0 else learned_filters,
                    out_channels = learned_filters,
                    kernel_size = kernel_size,
                    padding = kernel_size // 2, #preserve time dimension
                    bias=True
                )
            )

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        prod = 0
        out = []
        for arm, adj in adjacency.items():
            prod += layers[adj](x)
            #Something like this. Essentially we want to loop through our adjacency list and apply the 
            # support for each armature in the mapping per joint and average them over the number of adjacent elements.
        
        return out
        

