"""Core data types for RL algorithms."""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass(frozen=True)
class Transition:
    """Single environment transition.

    Attributes:
        observation: Current observation tensor.
        action: Action taken.
        reward: Reward received.
        next_observation: Next observation tensor.
        done: Episode termination flag.
        log_prob: Log probability of the action under the policy.
        value: Value estimate at current state.
    """
    observation: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: torch.Tensor
    done: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor


@dataclass(frozen=True)
class RolloutBatch:
    """Batched transitions for algorithm updates.

    Attributes:
        observations: Batch of observations [B, obs_dim] or [B, T, obs_dim].
        actions: Batch of actions [B, action_dim] or [B, T, action_dim].
        rewards: Batch of rewards [B] or [B, T].
        dones: Batch of done flags [B] or [B, T].
        log_probs: Log probabilities from rollout collection [B] or [B, T].
        values: Value estimates from rollout collection [B] or [B, T].
        advantages: Computed advantages [B] or [B, T].
        returns: Computed returns [B] or [B, T].
    """
    observations: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor

    def flatten(self) -> "RolloutBatch":
        """Flatten batch and time dimensions if present."""
        if self.observations.dim() == 2:
            return self

        B, T = self.observations.shape[:2]
        return RolloutBatch(
            observations=self.observations.reshape(B * T, -1),
            actions=self.actions.reshape(B * T, -1),
            rewards=self.rewards.reshape(B * T),
            dones=self.dones.reshape(B * T),
            log_probs=self.log_probs.reshape(B * T),
            values=self.values.reshape(B * T),
            advantages=self.advantages.reshape(B * T),
            returns=self.returns.reshape(B * T),
        )

    def to(self, device: torch.device) -> "RolloutBatch":
        """Move all tensors to specified device."""
        return RolloutBatch(
            observations=self.observations.to(device),
            actions=self.actions.to(device),
            rewards=self.rewards.to(device),
            dones=self.dones.to(device),
            log_probs=self.log_probs.to(device),
            values=self.values.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
        )


@dataclass(frozen=True)
class MotionState:
    """Motion observation for motion-conditioned RL.

    Attributes:
        joint_positions: World-space joint positions [T, J, 3] or [B, T, J, 3].
        joint_rotations: Joint rotations as quaternions (x,y,z,w) [T, J, 4] or [B, T, J, 4].
        joint_velocities: Joint velocities [T, J, 3] or [B, T, J, 3].
        root_position: Root position [T, 3] or [B, T, 3].
        root_velocity: Root velocity [T, 3] or [B, T, 3].
    """
    joint_positions: torch.Tensor
    joint_rotations: torch.Tensor
    joint_velocities: torch.Tensor
    root_position: torch.Tensor
    root_velocity: torch.Tensor

    def to(self, device: torch.device) -> "MotionState":
        """Move all tensors to specified device."""
        return MotionState(
            joint_positions=self.joint_positions.to(device),
            joint_rotations=self.joint_rotations.to(device),
            joint_velocities=self.joint_velocities.to(device),
            root_position=self.root_position.to(device),
            root_velocity=self.root_velocity.to(device),
        )

    def flatten_features(self) -> torch.Tensor:
        """Flatten all state components into a single feature tensor."""
        return torch.cat([
            self.joint_positions.flatten(start_dim=-2),
            self.joint_rotations.flatten(start_dim=-2),
            self.joint_velocities.flatten(start_dim=-2),
            self.root_position,
            self.root_velocity,
        ], dim=-1)


@dataclass(frozen=True)
class PolicyOutput:
    """Output from policy network forward pass.

    Attributes:
        action_mean: Mean of action distribution [B, action_dim].
        action_std: Standard deviation of action distribution [B, action_dim].
        log_prob: Log probability of sampled/given action [B].
        entropy: Entropy of action distribution [B].
        value: Value estimate (if using shared network) [B], optional.
    """
    action_mean: torch.Tensor
    action_std: torch.Tensor
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: Optional[torch.Tensor] = None
