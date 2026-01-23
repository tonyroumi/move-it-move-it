"""Entropy computation utilities."""

import torch
import math


class Entropy:
    """Static methods for entropy computation."""

    @staticmethod
    def gaussian(std: torch.Tensor) -> torch.Tensor:
        """Compute entropy of a Gaussian distribution.

        Args:
            std: Standard deviation tensor [B, action_dim].

        Returns:
            Entropy summed over action dimensions [B].
        """
        return 0.5 * (1.0 + math.log(2 * math.pi) + 2 * torch.log(std)).sum(dim=-1)

    @staticmethod
    def categorical(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Compute entropy of a categorical distribution.

        Args:
            probs: Probability tensor [B, num_actions].
            eps: Small constant for numerical stability.

        Returns:
            Entropy [B].
        """
        return -(probs * torch.log(probs + eps)).sum(dim=-1)

    @staticmethod
    def from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
        """Compute entropy from log probabilities (categorical).

        Args:
            log_probs: Log probability tensor [B, num_actions].

        Returns:
            Entropy [B].
        """
        probs = torch.exp(log_probs)
        return -(probs * log_probs).sum(dim=-1)
