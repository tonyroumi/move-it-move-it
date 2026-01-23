"""Policy gradient loss functions."""

import torch
from typing import Tuple, Dict


class PolicyLoss:
    """Static methods for policy gradient losses."""

    @staticmethod
    def clipped_surrogate(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        clip_epsilon: float,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute PPO clipped surrogate objective.

        Args:
            log_probs: Current log probabilities [B].
            old_log_probs: Old log probabilities from rollout [B].
            advantages: Normalized advantages [B].
            clip_epsilon: Clipping parameter.

        Returns:
            loss: Negative of clipped objective (for minimization).
            metrics: Dictionary containing clip fraction and approx KL.
        """
        ratio = torch.exp(log_probs - old_log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages

        loss = -torch.min(surr1, surr2).mean()

        with torch.no_grad():
            clip_fraction = ((ratio - 1.0).abs() > clip_epsilon).float().mean()
            approx_kl = (old_log_probs - log_probs).mean()

        metrics = {
            "clip_fraction": clip_fraction,
            "approx_kl": approx_kl,
            "ratio_mean": ratio.mean(),
        }

        return loss, metrics

    @staticmethod
    def vanilla_pg(
        log_probs: torch.Tensor,
        advantages: torch.Tensor,
    ) -> torch.Tensor:
        """Compute vanilla policy gradient loss.

        Args:
            log_probs: Log probabilities of actions [B].
            advantages: Advantages [B].

        Returns:
            Policy loss (negative expected advantage).
        """
        return -(log_probs * advantages.detach()).mean()

    @staticmethod
    def ppo_penalty(
        log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        kl_coef: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute PPO with KL penalty (adaptive KL).

        Args:
            log_probs: Current log probabilities [B].
            old_log_probs: Old log probabilities [B].
            advantages: Advantages [B].
            kl_coef: Current KL penalty coefficient.

        Returns:
            loss: Policy loss with KL penalty.
            kl: Approximate KL divergence.
        """
        ratio = torch.exp(log_probs - old_log_probs)
        surr = ratio * advantages

        kl = (old_log_probs - log_probs).mean()

        loss = -(surr.mean() - kl_coef * kl)

        return loss, kl
