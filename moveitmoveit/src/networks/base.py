from typing import List

import torch.nn as nn
import torch

from moveitmoveit.src.networks.utils import get_activation

class BaseMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_layers: List[int] = [256, 256],
        activation: str = "relu"
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layers = hidden_layers
        self.activation = activation

        layers = []
        dims = [in_channels] + hidden_layers
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(get_activation(activation))
        layers.append(nn.Linear(dims[-1], out_channels))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)