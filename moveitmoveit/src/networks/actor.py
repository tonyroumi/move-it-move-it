from typing import List, Iterator, Optional

import torch.nn as nn
import torch
from torch.distributions import Normal
from torch.nn import Parameter

from moveitmoveit.src.networks.base import BaseMLP

class GaussianActor(BaseMLP):
    """ Gaussian Actor Network. """
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int,
        hidden_layers: List[int] = [256, 256],
        activation: str = "relu",
        init_noise_std = 1.0,
        noise_std_type="scalar",
        ortho_init: bool = True
        ):
        super().__init__(in_channels, out_channels, hidden_layers, activation)

        self.noise_std_type = noise_std_type

        if self.noise_std_type == "fixed":
            self.register_buffer("std", torch.ones(self.out_channels) * init_noise_std)
        elif self.noise_std_type == "scalar":
            self.std = nn.Parameter(torch.ones(self.out_channels) * init_noise_std)
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(self.out_channels)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        
        #Action Distribution (populated in update_distribution)
        self.distribution = None
        #disable args validation for speedup
        Normal.set_default_validate_args(False)

        if ortho_init:
            self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        # Smaller gain on the output (mean) head for stable initial actions.
        mean_head = self.network[-1]
        nn.init.orthogonal_(mean_head.weight, gain=0.01)
        nn.init.zeros_(mean_head.bias)
    
    def update_distribution(self, obs: torch.Tensor):
        """Update the action distribution."""
        mean = super().forward(obs)
        if self.noise_std_type == "fixed":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std.expand_as(mean))
        self.distribution = Normal(mean, std)
    
    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        self.update_distribution(obs)
        if deterministic:
            action = self.distribution.mean
        else:
            action = self.distribution.sample()

        return action
    
    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """ Get log probabilities of actions. """
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    @property
    def action_mean(self) -> torch.Tensor: 
        """Mean of the action distribution."""
        return self.distribution.mean
    
    @property
    def action_std(self) -> torch.Tensor:
        """Standard deviation of the action distribution."""
        return self.distribution.stddev
    
    @property
    def entropy(self) -> torch.Tensor:
        """Entropy of the action distribution."""
        return self.distribution.entropy().sum(dim=-1)
    
    def parameters(self) -> Iterator[Parameter]:
        """Parameters of the network."""
        return self.network.parameters()