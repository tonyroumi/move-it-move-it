"""Advantage and return computation utilities."""

import torch
from typing import Tuple


class Advantage:
    """Static methods for advantage estimation."""

    @staticmethod
    def gae(
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float,
        gae_lambda: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Generalized Advantage Estimation.

        Args:
            rewards: Rewards tensor [B, T] or [T].
            values: Value estimates [B, T] or [T].
            dones: Done flags [B, T] or [T].
            next_value: Bootstrap value [B] or scalar.
            gamma: Discount factor.
            gae_lambda: GAE lambda parameter.

        Returns:
            advantages: Computed advantages, same shape as rewards.
            returns: Computed returns (advantages + values).
        """
        is_batched = rewards.dim() == 2

        if not is_batched:
            rewards = rewards.unsqueeze(0)
            values = values.unsqueeze(0)
            dones = dones.unsqueeze(0)
            next_value = next_value.unsqueeze(0) if next_value.dim() == 0 else next_value.unsqueeze(0)

        B, T = rewards.shape
        device = rewards.device
        advantages = torch.zeros(B, T, device=device)
        last_gae = torch.zeros(B, device=device)

        values_extended = torch.cat([values, next_value.unsqueeze(1)], dim=1)

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[:, t].float()
            delta = (
                rewards[:, t]
                + gamma * values_extended[:, t + 1] * next_non_terminal
                - values[:, t]
            )
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[:, t] = last_gae

        returns = advantages + values

        if not is_batched:
            advantages = advantages.squeeze(0)
            returns = returns.squeeze(0)

        return advantages, returns

    @staticmethod
    def n_step_returns(
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_value: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """Compute n-step returns.

        Args:
            rewards: Rewards tensor [B, T] or [T].
            dones: Done flags [B, T] or [T].
            next_value: Bootstrap value [B] or scalar.
            gamma: Discount factor.

        Returns:
            returns: Computed returns, same shape as rewards.
        """
        is_batched = rewards.dim() == 2

        if not is_batched:
            rewards = rewards.unsqueeze(0)
            dones = dones.unsqueeze(0)
            next_value = next_value.unsqueeze(0) if next_value.dim() == 0 else next_value.unsqueeze(0)

        B, T = rewards.shape
        device = rewards.device
        returns = torch.zeros(B, T, device=device)

        R = next_value.clone()

        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[:, t].float()
            R = rewards[:, t] + gamma * R * next_non_terminal
            returns[:, t] = R

        if not is_batched:
            returns = returns.squeeze(0)

        return returns

    @staticmethod
    def normalize(advantages: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        """Normalize advantages to zero mean and unit variance.

        Args:
            advantages: Advantages tensor of any shape.
            eps: Small constant for numerical stability.

        Returns:
            Normalized advantages.
        """
        return (advantages - advantages.mean()) / (advantages.std() + eps)
