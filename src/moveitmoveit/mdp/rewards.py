from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse

from moveitmoveit.mdp.commands import CommandIndex
from moveitmoveit.utils.transforms import heading_frame

if TYPE_CHECKING:
    from moveitmoveit.mdp.mdp import StepContext


class RewardTerm(NamedTuple):
    fn: Callable[..., torch.Tensor]
    defaults: dict[str, float]


REWARDS: dict[str, RewardTerm] = {}


def reward(**defaults: float):
    def register(fn):
        REWARDS[fn.__name__] = RewardTerm(fn, defaults)
        return fn

    return register


@reward(dof_scale=2.0, key_body_scale=5.0)
def motion_tracking(c: StepContext, dof_scale: torch.Tensor, key_body_scale: torch.Tensor) -> torch.Tensor:
    """Reward for matching target joint positions and relative key body positions."""
    dof_error = torch.sum((c.state.dof_positions - c.reference_state.dof_positions) ** 2, dim=-1)
    offsets = c.state.key_body_positions - c.state.root_positions.unsqueeze(1)
    ref_offsets = c.reference_state.key_body_positions - c.reference_state.root_positions.unsqueeze(1)
    key_body_error = torch.sum((offsets - ref_offsets).flatten(1) ** 2, dim=-1)
    return torch.exp(-dof_scale * dof_error - key_body_scale * key_body_error)


@reward(dof_scale=2.0, key_body_scale=5.0)
def heading_motion_tracking(c: StepContext, dof_scale: torch.Tensor, key_body_scale: torch.Tensor) -> torch.Tensor:
    """Reward for matching target joint positions and relative key body positions in the heading frame."""
    dof_error = torch.sum((c.state.dof_positions - c.reference_state.dof_positions) ** 2, dim=-1)
    offsets = heading_frame(
        c.state.key_body_positions - c.state.root_positions.unsqueeze(1),
        c.state.root_rotations
    )
    ref_offsets = heading_frame(
        c.reference_state.key_body_positions - c.reference_state.root_positions.unsqueeze(1),
        c.reference_state.root_rotations
    )
    key_body_error = torch.sum((offsets - ref_offsets).flatten(1) ** 2, dim=-1)
    return torch.exp(-dof_scale * dof_error - key_body_scale * key_body_error)


@reward(scale=2.0)
def lin_vel_tracking(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for matching commanded linear velocity (xy)."""
    local_lin_vel_xy = quat_apply_inverse(c.state.root_rotations, c.state.root_linear_velocities)[:, :2]
    commanded_lin_vel = c.commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1]
    return torch.exp(-scale * torch.sum((local_lin_vel_xy - commanded_lin_vel) ** 2, dim=-1))


@reward(scale=2.0)
def yaw_vel_tracking(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for matching commanded yaw rate."""
    local_yaw_rate = quat_apply_inverse(c.state.root_rotations, c.state.root_angular_velocities)[:, 2]
    return torch.exp(-scale * (local_yaw_rate - c.commands[:, CommandIndex.YAW]) ** 2)


@reward(scale=0.5)
def target_hit(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for being close to the goal position (xy)."""
    root_xy = c.state.root_positions[:, :2]
    goal = c.commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1]
    return torch.exp(-scale * torch.sum((root_xy - goal) ** 2, dim=-1))


@reward(scale=1.0)
def action_rate_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for smooth actions (small change from previous actions)."""
    return torch.exp(-scale * torch.sum((c.actions - c.previous_actions) ** 2, dim=-1))


@reward(scale=1.0e-4)
def joint_acc(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for low joint accelerations."""
    return torch.exp(-scale * torch.sum(c.state.dof_accelerations**2, dim=-1))


@reward(scale=1.0e-2)
def joint_vel(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for low joint velocities."""
    return torch.exp(-scale * torch.sum(c.state.dof_velocities**2, dim=-1))


@reward(scale=1.0)
def progress(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for reducing distance to the goal."""
    root_xy = c.state.root_positions[:, :2]
    goal = c.commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1]
    distance = torch.norm(goal - root_xy, dim=-1)
    previous = c.memory.get("goal_distance", distance)
    previous = torch.where(torch.isnan(previous), distance, previous)
    c.memory["goal_distance"] = distance
    return scale * (previous - distance)


@reward(scale=4.0)
def root_lin_vel_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for low root linear velocity."""
    return torch.exp(-scale * torch.sum(c.state.root_linear_velocities**2, dim=-1))


@reward(scale=2.0)
def root_ang_vel_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for low root angular velocity."""
    return torch.exp(-scale * torch.sum(c.state.root_angular_velocities**2, dim=-1))


@reward(scale=10.0)
def upright(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for staying upright."""
    up_axis = torch.zeros_like(c.state.root_positions)
    up_axis[:, 2] = 1.0
    tilt = 1.0 - quat_apply(c.state.root_rotations, up_axis)[:, 2]
    return torch.exp(-scale * tilt)


@reward(scale=20.0)
def support_centered(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Reward for keeping the root centered over the feet."""
    feet_midpoint = c.state.key_body_positions[:, -2:, :2].mean(dim=1)
    root_xy = c.state.root_positions[:, :2]
    return torch.exp(-scale * torch.sum((root_xy - feet_midpoint) ** 2, dim=-1))


@reward(scale=2.0)
def reference_pose(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Joint positions vs. the reference pose."""
    error = torch.sum((c.state.dof_positions - c.reference_state.dof_positions) ** 2, dim=-1)
    return torch.exp(-scale * error)