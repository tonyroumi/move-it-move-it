"""Critic (value function) networks."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, List


@dataclass(frozen=True)
class CriticConfig:
    """Configuration for critic network.

    Attributes:
        input_dim: Dimension of observation input.
        hidden_dims: Dimensions of hidden layers.
        activation: Activation function name.
        ortho_init: Whether to use orthogonal initialization.
    """
    input_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "elu"
    ortho_init: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "CriticConfig":
        """Construct from dictionary config."""
        if "hidden_dims" in cfg and isinstance(cfg["hidden_dims"], list):
            cfg = dict(cfg)
            cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class ValueCritic(nn.Module):
    """Value function critic.

    Outputs scalar value estimate for given observation.
    """

    def __init__(self, config: CriticConfig):
        super().__init__()
        self.config = config

        layers: List[nn.Module] = []
        prev_dim = config.input_dim
        activation = self._get_activation(config.activation)

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))

        self.network = nn.Sequential(*layers)

        if config.ortho_init:
            self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization."""
        for i, module in enumerate(self.network):
            if isinstance(module, nn.Linear):
                if i == len(self.network) - 1:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                else:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "elu": nn.ELU(),
            "tanh": nn.Tanh(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.ELU())

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass returning value estimate.

        Args:
            obs: Observation tensor [B, obs_dim].

        Returns:
            Value estimate [B, 1].
        """
        return self.network(obs)

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        """Get value estimate as scalar.

        Args:
            obs: Observation tensor [B, obs_dim].

        Returns:
            Value estimate [B].
        """
        return self.forward(obs).squeeze(-1)
