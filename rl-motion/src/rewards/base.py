"""Base reward module protocol."""

from typing import Protocol, Optional
import torch

from motion.src.core.types import MotionState


class RewardModule(Protocol):
    """Protocol defining the interface for reward computation modules.

    All reward modules should implement this interface to allow
    composition and interchangeability.
    """

    def compute(
        self,
        current_state: MotionState,
        target_state: Optional[MotionState],
        action: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Compute reward given current and target states.

        Args:
            current_state: Current motion state from simulation/policy.
            target_state: Reference/target motion state (optional for some rewards).
            action: Action taken (optional, for action-dependent rewards).

        Returns:
            Scalar or batched reward tensor.
        """
        ...

    def reset(self) -> None:
        """Reset any internal state (e.g., for episode boundaries)."""
        ...
