"""Value function loss computations."""

import torch
import torch.nn.functional as F


class ValueLoss:
    """Static methods for value function losses."""

    @staticmethod
    def mse(
        values: torch.Tensor,
        returns: torch.Tensor,
    ) -> torch.Tensor:
        """Compute mean squared error value loss.

        Args:
            values: Value predictions [B].
            returns: Target returns [B].

        Returns:
            MSE loss.
        """
        return F.mse_loss(values, returns)

    @staticmethod
    def huber(
        values: torch.Tensor,
        returns: torch.Tensor,
        delta: float = 1.0,
    ) -> torch.Tensor:
        """Compute Huber loss for value function.

        Args:
            values: Value predictions [B].
            returns: Target returns [B].
            delta: Threshold for switching between L1 and L2.

        Returns:
            Huber loss.
        """
        return F.smooth_l1_loss(values, returns, beta=delta)

    @staticmethod
    def clipped(
        values: torch.Tensor,
        old_values: torch.Tensor,
        returns: torch.Tensor,
        clip_epsilon: float,
    ) -> torch.Tensor:
        """Compute clipped value loss (PPO-style).

        Args:
            values: Current value predictions [B].
            old_values: Old value predictions from rollout [B].
            returns: Target returns [B].
            clip_epsilon: Clipping parameter.

        Returns:
            Clipped value loss.
        """
        value_clipped = old_values + torch.clamp(
            values - old_values, -clip_epsilon, clip_epsilon
        )

        loss_unclipped = (values - returns) ** 2
        loss_clipped = (value_clipped - returns) ** 2

        return 0.5 * torch.max(loss_unclipped, loss_clipped).mean()
