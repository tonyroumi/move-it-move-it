"""Motion discriminator network for AMP-style rewards."""

import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Tuple, List


@dataclass(frozen=True)
class DiscriminatorConfig:
    """Configuration for motion discriminator network.

    Attributes:
        input_dim: Dimension of flattened motion state input.
        hidden_dims: Dimensions of hidden layers.
        activation: Activation function name.
        use_spectral_norm: Whether to apply spectral normalization.
        dropout: Dropout probability (0 to disable).
    """
    input_dim: int
    hidden_dims: Tuple[int, ...] = (1024, 512)
    activation: str = "relu"
    use_spectral_norm: bool = True
    dropout: float = 0.0

    @classmethod
    def from_dict(cls, cfg: dict) -> "DiscriminatorConfig":
        """Construct from dictionary config."""
        if "hidden_dims" in cfg and isinstance(cfg["hidden_dims"], list):
            cfg = dict(cfg)
            cfg["hidden_dims"] = tuple(cfg["hidden_dims"])
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class MotionDiscriminator(nn.Module):
    """Discriminator for distinguishing real vs generated motion.

    Used in AMP (Adversarial Motion Prior) to provide style rewards.
    """

    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.config = config

        layers: List[nn.Module] = []
        prev_dim = config.input_dim

        for hidden_dim in config.hidden_dims:
            linear = nn.Linear(prev_dim, hidden_dim)

            if config.use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)

            layers.append(linear)
            layers.append(self._get_activation(config.activation))

            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

            prev_dim = hidden_dim

        output_layer = nn.Linear(prev_dim, 1)
        if config.use_spectral_norm:
            output_layer = nn.utils.spectral_norm(output_layer)

        layers.append(output_layer)

        self.network = nn.Sequential(*layers)

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation module by name."""
        activations = {
            "relu": nn.ReLU(),
            "leaky_relu": nn.LeakyReLU(0.2),
            "elu": nn.ELU(),
            "silu": nn.SiLU(),
        }
        return activations.get(name, nn.LeakyReLU(0.2))

    def forward(self, motion_features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            motion_features: Flattened motion features [B, feature_dim].

        Returns:
            Discriminator logits [B, 1].
        """
        return self.network(motion_features)

    def get_reward(self, motion_features: torch.Tensor) -> torch.Tensor:
        """Compute reward from discriminator output.

        Uses sigmoid activation to convert logits to [0, 1] reward.

        Args:
            motion_features: Flattened motion features [B, feature_dim].

        Returns:
            Reward values [B].
        """
        logits = self.forward(motion_features)
        return torch.sigmoid(logits).squeeze(-1)

    def compute_gradient_penalty(
        self,
        real: torch.Tensor,
        fake: torch.Tensor,
    ) -> torch.Tensor:
        """Compute gradient penalty for discriminator regularization.

        Args:
            real: Real motion features [B, feature_dim].
            fake: Fake/generated motion features [B, feature_dim].

        Returns:
            Gradient penalty scalar.
        """
        alpha = torch.rand(real.size(0), 1, device=real.device)
        interpolated = alpha * real + (1 - alpha) * fake.detach()
        interpolated.requires_grad_(True)

        pred = self.forward(interpolated)

        gradients = torch.autograd.grad(
            outputs=pred,
            inputs=interpolated,
            grad_outputs=torch.ones_like(pred),
            create_graph=True,
            retain_graph=True,
        )[0]

        gradient_norm = gradients.norm(2, dim=1)
        return ((gradient_norm - 1) ** 2).mean()
