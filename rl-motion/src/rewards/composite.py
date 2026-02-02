"""Composite reward combining multiple reward modules."""

import torch
from typing import Dict, Optional

from motion.src.core.types import MotionState
from motion.src.rewards.base import RewardModule


class CompositeReward:
    """Combines multiple reward modules with configurable weights.

    Allows mixing task rewards, motion matching, and discriminator rewards
    into a single unified reward signal.
    """

    def __init__(
        self,
        modules: Dict[str, RewardModule],
        weights: Dict[str, float],
    ):
        """Initialize composite reward.

        Args:
            modules: Dictionary mapping names to reward modules.
            weights: Dictionary mapping names to weights.

        Raises:
            ValueError: If weight keys don't match module keys.
        """
        if set(modules.keys()) != set(weights.keys()):
            raise ValueError(
                f"Weight keys {set(weights.keys())} must match "
                f"module keys {set(modules.keys())}"
            )

        self.modules = modules
        self.weights = weights

    def compute(
        self,
        current_state: MotionState,
        target_state: Optional[MotionState],
        action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute weighted sum of all reward modules.

        Args:
            current_state: Current motion state.
            target_state: Target motion state (optional).
            action: Action taken (optional).

        Returns:
            Combined reward.
        """
        total_reward = torch.tensor(0.0, device=current_state.joint_positions.device)

        for name, module in self.modules.items():
            reward = module.compute(current_state, target_state, action)
            total_reward = total_reward + self.weights[name] * reward

        return total_reward

    def compute_with_breakdown(
        self,
        current_state: MotionState,
        target_state: Optional[MotionState],
        action: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute rewards with per-module breakdown for logging.

        Args:
            current_state: Current motion state.
            target_state: Target motion state (optional).
            action: Action taken (optional).

        Returns:
            total_reward: Combined reward.
            breakdown: Dictionary of individual module rewards (unweighted).
        """
        breakdown: Dict[str, torch.Tensor] = {}
        total_reward = torch.tensor(0.0, device=current_state.joint_positions.device)

        for name, module in self.modules.items():
            reward = module.compute(current_state, target_state, action)
            breakdown[name] = reward
            total_reward = total_reward + self.weights[name] * reward

        return total_reward, breakdown

    def reset(self) -> None:
        """Reset all reward modules."""
        for module in self.modules.values():
            module.reset()

    def add_module(self, name: str, module: RewardModule, weight: float) -> None:
        """Add a new reward module.

        Args:
            name: Name for the module.
            module: Reward module instance.
            weight: Weight for the module.
        """
        self.modules[name] = module
        self.weights[name] = weight

    def remove_module(self, name: str) -> None:
        """Remove a reward module.

        Args:
            name: Name of module to remove.
        """
        if name in self.modules:
            del self.modules[name]
            del self.weights[name]

    def set_weight(self, name: str, weight: float) -> None:
        """Update weight for a module.

        Args:
            name: Module name.
            weight: New weight value.

        Raises:
            KeyError: If module name not found.
        """
        if name not in self.weights:
            raise KeyError(f"Module '{name}' not found in composite reward")
        self.weights[name] = weight
