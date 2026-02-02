"""Discriminator-based reward computation for AMP."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from motion.src.core.types import MotionState
from motion.src.networks.discriminator import MotionDiscriminator


@dataclass(frozen=True)
class DiscriminatorRewardConfig:
    """Configuration for discriminator-based reward.

    Attributes:
        reward_scale: Scale factor for reward.
        use_sigmoid: Convert logits to [0,1] using sigmoid.
        gradient_penalty_coef: Coefficient for gradient penalty in discriminator loss.
        use_lsgan: Use LSGAN loss instead of standard GAN loss.
    """
    reward_scale: float = 1.0
    use_sigmoid: bool = True
    gradient_penalty_coef: float = 5.0
    use_lsgan: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "DiscriminatorRewardConfig":
        """Construct from dictionary config."""
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class DiscriminatorReward:
    """AMP-style discriminator reward.

    The discriminator learns to distinguish real motion from policy-generated
    motion. The reward is based on how "real" the discriminator thinks the
    motion is.
    """

    def __init__(
        self,
        discriminator: MotionDiscriminator,
        config: DiscriminatorRewardConfig,
    ):
        self.discriminator = discriminator
        self.config = config

    def compute(
        self,
        current_state: MotionState,
        target_state: Optional[MotionState] = None,
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute discriminator-based reward.

        Args:
            current_state: Current motion state to evaluate.
            target_state: Unused (discriminator doesn't need reference).
            action: Unused.

        Returns:
            Reward tensor based on discriminator output.
        """
        motion_features = current_state.flatten_features()

        with torch.no_grad():
            disc_output = self.discriminator(motion_features)

        if self.config.use_sigmoid:
            reward = torch.sigmoid(disc_output).squeeze(-1)
        else:
            reward = -torch.log(
                torch.clamp(1.0 - torch.sigmoid(disc_output), min=1e-8)
            ).squeeze(-1)

        return reward * self.config.reward_scale

    def compute_discriminator_loss(
        self,
        real_states: MotionState,
        fake_states: MotionState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute discriminator training loss.

        Args:
            real_states: Motion states from reference dataset.
            fake_states: Motion states from policy rollout.

        Returns:
            loss: Total discriminator loss.
            metrics: Dictionary of loss components.
        """
        real_features = real_states.flatten_features()
        fake_features = fake_states.flatten_features()

        real_pred = self.discriminator(real_features)
        fake_pred = self.discriminator(fake_features.detach())

        if self.config.use_lsgan:
            loss_real = F.mse_loss(real_pred, torch.ones_like(real_pred))
            loss_fake = F.mse_loss(fake_pred, torch.zeros_like(fake_pred))
        else:
            loss_real = F.binary_cross_entropy_with_logits(
                real_pred, torch.ones_like(real_pred)
            )
            loss_fake = F.binary_cross_entropy_with_logits(
                fake_pred, torch.zeros_like(fake_pred)
            )

        disc_loss = 0.5 * (loss_real + loss_fake)

        gp = torch.tensor(0.0, device=real_features.device)
        if self.config.gradient_penalty_coef > 0:
            gp = self.discriminator.compute_gradient_penalty(real_features, fake_features)
            disc_loss = disc_loss + self.config.gradient_penalty_coef * gp

        metrics = {
            "loss_real": loss_real,
            "loss_fake": loss_fake,
            "gradient_penalty": gp,
            "real_pred_mean": torch.sigmoid(real_pred).mean(),
            "fake_pred_mean": torch.sigmoid(fake_pred).mean(),
        }

        return disc_loss, metrics

    def reset(self) -> None:
        """Reset internal state (no state for this reward)."""
        pass
