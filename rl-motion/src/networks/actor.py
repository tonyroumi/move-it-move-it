"""Actor (policy) networks."""

import torch
import torch.nn as nn
from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Tuple, List


@dataclass(frozen=True)
class ActorConfig:
    """Configuration for actor network.

    Attributes:
        input_dim: Dimension of observation input.
        output_dim: Dimension of action output.
        hidden_dims: Dimensions of hidden layers.
        activation: Activation function name.
        log_std_init: Initial value for log standard deviation.
        log_std_min: Minimum log standard deviation.
        log_std_max: Maximum log standard deviation.
        ortho_init: Whether to use orthogonal initialization.
    """
    input_dim: int
    output_dim: int
    hidden_dims: Tuple[int, ...] = (256, 256)
    activation: str = "elu"
    log_std_init: float = -1.0
    log_std_min: float = -5.0
    log_std_max: float = 2.0
    ortho_init: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "ActorConfig":
        """Construct from dictionary config."""
        if "hidden_dims" in cfg and isinstance(cfg["hidden_dims"], list):
            cfg = dict(cfg)
            cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class GaussianActor(nn.Module):
    """Gaussian policy network.

    Outputs a Normal distribution over continuous actions.
    """

    def __init__(self, config: ActorConfig):
        super().__init__()
        self.config = config

        layers: List[nn.Module] = []
        prev_dim = config.input_dim
        activation = self._get_activation(config.activation)

        for hidden_dim in config.hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(activation)
            prev_dim = hidden_dim

        self.backbone = nn.Sequential(*layers)
        self.mean_head = nn.Linear(prev_dim, config.output_dim)
        self.log_std = nn.Parameter(
            torch.ones(config.output_dim) * config.log_std_init
        )

        if config.ortho_init:
            self._init_weights()

    def _init_weights(self) -> None:
        """Apply orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=1.0)
                nn.init.zeros_(module.bias)

        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)

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

    def forward(self, obs: torch.Tensor) -> Normal:
        """Forward pass returning action distribution.

        Args:
            obs: Observation tensor [B, obs_dim].

        Returns:
            Normal distribution over actions.
        """
        features = self.backbone(obs)
        mean = self.mean_head(features)

        log_std = torch.clamp(
            self.log_std, self.config.log_std_min, self.config.log_std_max
        )
        std = torch.exp(log_std)

        return Normal(mean, std.expand_as(mean))

    def get_action(
        self, obs: torch.Tensor, deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability and entropy.

        Args:
            obs: Observation tensor [B, obs_dim].
            deterministic: If True, return mean action.

        Returns:
            action: Sampled or mean action [B, action_dim].
            log_prob: Log probability of action [B].
            entropy: Entropy of distribution [B].
        """
        dist = self.forward(obs)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Evaluate log probability and entropy for given actions.

        Args:
            obs: Observation tensor [B, obs_dim].
            actions: Action tensor [B, action_dim].

        Returns:
            log_prob: Log probability of actions [B].
            entropy: Entropy of distribution [B].
        """
        dist = self.forward(obs)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy
