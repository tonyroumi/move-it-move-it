"""Motion matching reward computation."""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional

from motion.src.core.types import MotionState


@dataclass(frozen=True)
class MotionMatchingConfig:
    """Configuration for motion matching reward.

    Attributes:
        position_weight: Weight for joint position matching.
        rotation_weight: Weight for joint rotation matching.
        velocity_weight: Weight for joint velocity matching.
        root_position_weight: Weight for root position matching.
        root_velocity_weight: Weight for root velocity matching.
        scale: Overall reward scale.
        use_exp_reward: Use exponential reward (exp(-error)) vs linear.
    """
    position_weight: float = 1.0
    rotation_weight: float = 0.5
    velocity_weight: float = 0.1
    root_position_weight: float = 1.0
    root_velocity_weight: float = 0.1
    scale: float = 1.0
    use_exp_reward: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "MotionMatchingConfig":
        """Construct from dictionary config."""
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})


class MotionMatchingReward:
    """Computes reward based on matching reference motion.

    Implements various motion similarity metrics including joint positions,
    rotations (quaternions), and velocities.
    """

    def __init__(self, config: MotionMatchingConfig):
        self.config = config

    def compute(
        self,
        current_state: MotionState,
        target_state: Optional[MotionState],
        action: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute motion matching reward.

        Args:
            current_state: Current motion state.
            target_state: Target/reference motion state.
            action: Unused for motion matching.

        Returns:
            Reward tensor.
        """
        if target_state is None:
            raise ValueError("Motion matching requires a target state")

        rewards = {}
        cfg = self.config

        if cfg.position_weight > 0:
            pos_error = self._position_error(
                current_state.joint_positions, target_state.joint_positions
            )
            rewards["position"] = self._error_to_reward(pos_error, cfg.position_weight)

        if cfg.rotation_weight > 0:
            rot_error = self._quaternion_error(
                current_state.joint_rotations, target_state.joint_rotations
            )
            rewards["rotation"] = self._error_to_reward(rot_error, cfg.rotation_weight)

        if cfg.velocity_weight > 0:
            vel_error = self._position_error(
                current_state.joint_velocities, target_state.joint_velocities
            )
            rewards["velocity"] = self._error_to_reward(vel_error, cfg.velocity_weight)

        if cfg.root_position_weight > 0:
            root_pos_error = self._position_error(
                current_state.root_position, target_state.root_position
            )
            rewards["root_position"] = self._error_to_reward(
                root_pos_error, cfg.root_position_weight
            )

        if cfg.root_velocity_weight > 0:
            root_vel_error = self._position_error(
                current_state.root_velocity, target_state.root_velocity
            )
            rewards["root_velocity"] = self._error_to_reward(
                root_vel_error, cfg.root_velocity_weight
            )

        if not rewards:
            return torch.zeros(1, device=current_state.joint_positions.device)

        total_reward = sum(rewards.values()) / len(rewards)
        return total_reward * cfg.scale

    def compute_with_breakdown(
        self,
        current_state: MotionState,
        target_state: Optional[MotionState],
        action: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute reward with per-component breakdown for logging.

        Args:
            current_state: Current motion state.
            target_state: Target/reference motion state.
            action: Unused.

        Returns:
            total_reward: Combined reward.
            breakdown: Dictionary of component rewards.
        """
        if target_state is None:
            raise ValueError("Motion matching requires a target state")

        breakdown = {}
        cfg = self.config

        if cfg.position_weight > 0:
            pos_error = self._position_error(
                current_state.joint_positions, target_state.joint_positions
            )
            breakdown["position"] = self._error_to_reward(pos_error, cfg.position_weight)

        if cfg.rotation_weight > 0:
            rot_error = self._quaternion_error(
                current_state.joint_rotations, target_state.joint_rotations
            )
            breakdown["rotation"] = self._error_to_reward(rot_error, cfg.rotation_weight)

        if cfg.velocity_weight > 0:
            vel_error = self._position_error(
                current_state.joint_velocities, target_state.joint_velocities
            )
            breakdown["velocity"] = self._error_to_reward(vel_error, cfg.velocity_weight)

        if cfg.root_position_weight > 0:
            root_pos_error = self._position_error(
                current_state.root_position, target_state.root_position
            )
            breakdown["root_position"] = self._error_to_reward(
                root_pos_error, cfg.root_position_weight
            )

        if cfg.root_velocity_weight > 0:
            root_vel_error = self._position_error(
                current_state.root_velocity, target_state.root_velocity
            )
            breakdown["root_velocity"] = self._error_to_reward(
                root_vel_error, cfg.root_velocity_weight
            )

        if not breakdown:
            device = current_state.joint_positions.device
            return torch.zeros(1, device=device), {}

        total_reward = sum(breakdown.values()) / len(breakdown) * cfg.scale
        return total_reward, breakdown

    def _error_to_reward(self, error: torch.Tensor, weight: float) -> torch.Tensor:
        """Convert error to reward using configured transformation."""
        if self.config.use_exp_reward:
            return torch.exp(-weight * error)
        else:
            return torch.clamp(1.0 - weight * error, min=0.0)

    @staticmethod
    def _position_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute mean squared position error."""
        return F.mse_loss(pred, target, reduction="none").sum(dim=-1).mean()

    @staticmethod
    def _quaternion_error(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        """Compute quaternion distance (angle between rotations).

        Uses the formula: angle = 2 * arccos(|q1 . q2|)

        Args:
            q1: First quaternion tensor [..., 4] (x, y, z, w format).
            q2: Second quaternion tensor [..., 4] (x, y, z, w format).

        Returns:
            Mean angular error.
        """
        dot = (q1 * q2).sum(dim=-1).abs()
        dot = torch.clamp(dot, -1.0, 1.0)
        angle = 2.0 * torch.acos(dot)
        return angle.mean()

    def reset(self) -> None:
        """Reset internal state (no state for this reward)."""
        pass
