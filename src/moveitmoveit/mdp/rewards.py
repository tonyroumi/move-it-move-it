from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch

from isaaclab.utils.math import quat_apply, quat_apply_inverse, yaw_quat

from moveitmoveit.motion.manifest import CommandIndex
from moveitmoveit.utils.transforms import world_to_heading_frame_xy

if TYPE_CHECKING:
    from moveitmoveit.mdp.composer import StepContext


class RewardTerm(NamedTuple):
    fn: Callable[..., torch.Tensor]
    defaults: dict[str, float]


REWARDS: dict[str, RewardTerm] = {}


def reward(**defaults: float):
    def register(fn):
        REWARDS[fn.__name__] = RewardTerm(fn, defaults)
        return fn

    return register


@reward(dof_scale=2.0, key_body_scale=5.0, dof_vel_scale=0.05, root_vel_scale=2.0, root_ang_vel_scale=0.5)
def motion_tracking(
    c: StepContext,
    dof_scale: float = 2.0,
    key_body_scale: float = 5.0,
    dof_vel_scale: float = 0.05,
    root_vel_scale: float = 2.0,
    root_ang_vel_scale: float = 0.5,
    dof_weight: float = 0.25,
    key_body_weight: float = 0.20,
    dof_vel_weight: float = 0.20,
    root_vel_weight: float = 0.25,
    root_ang_vel_weight: float = 0.10,
) -> torch.Tensor:
    """Track reference pose and motion dynamics."""
    state = c.state
    targets = c.targets

    def tracking_reward(actual, target, scale):
        error = (actual - target).square().mean(dim=-1)
        return torch.exp(-scale * error)

    offsets = state.key_body_positions - state.root_positions.unsqueeze(1)
    ref_offsets = targets.key_body_positions - targets.root_positions.unsqueeze(1)

    return (
        dof_weight * tracking_reward(
            state.dof_positions, targets.dof_positions, dof_scale
        )
        + key_body_weight * tracking_reward(
            offsets.flatten(1), ref_offsets.flatten(1), key_body_scale
        )
        + dof_vel_weight * tracking_reward(
            state.dof_velocities, targets.dof_velocities, dof_vel_scale
        )
        + root_vel_weight * tracking_reward(
            state.root_linear_velocities,
            targets.root_linear_velocities,
            root_vel_scale,
        )
        + root_ang_vel_weight * tracking_reward(
            state.root_angular_velocities,
            targets.root_angular_velocities,
            root_ang_vel_scale,
        )
    )


@reward(dof_scale=2.0, key_body_scale=5.0, dof_vel_scale=0.05, root_vel_scale=2.0, root_ang_vel_scale=0.5,)
def heading_motion_tracking(
    c: StepContext,
    dof_scale: float = 2.0,
    key_body_scale: float = 5.0,
    dof_vel_scale: float = 0.05,
    root_vel_scale: float = 2.0,
    root_ang_vel_scale: float = 0.5,
    dof_weight: float = 0.25,
    key_body_weight: float = 0.20,
    dof_vel_weight: float = 0.20,
    root_vel_weight: float = 0.25,
    root_ang_vel_weight: float = 0.10,
) -> torch.Tensor:
    """Track reference pose and velocities in each root's heading frame."""
    state = c.state
    targets = c.targets

    def track(actual, target, scale):
        error = (actual - target).square().mean(dim=-1)
        return torch.exp(-scale * error)

    # Root-relative key-body positions in heading frames
    offsets = _heading_frame_offsets(
        state.key_body_positions,
        state.root_positions,
        state.root_rotations,
    )
    ref_offsets = _heading_frame_offsets(
        targets.key_body_positions,
        targets.root_positions,
        targets.root_rotations,
    )

    # Root velocities in heading frames
    root_vel = _heading_frame_vectors(
        state.root_linear_velocities,
        state.root_rotations,
    )
    ref_root_vel = _heading_frame_vectors(
        targets.root_linear_velocities,
        targets.root_rotations,
    )

    root_ang_vel = _heading_frame_vectors(
        state.root_angular_velocities,
        state.root_rotations,
    )
    ref_root_ang_vel = _heading_frame_vectors(
        targets.root_angular_velocities,
        targets.root_rotations,
    )

    return (
        dof_weight * track(
            state.dof_positions, targets.dof_positions, dof_scale
        )
        + key_body_weight * track(
            offsets.flatten(1), ref_offsets.flatten(1), key_body_scale
        )
        + dof_vel_weight * track(
            state.dof_velocities, targets.dof_velocities, dof_vel_scale
        )
        + root_vel_weight * track(
            root_vel, ref_root_vel, root_vel_scale
        )
        + root_ang_vel_weight * track(
            root_ang_vel, ref_root_ang_vel, root_ang_vel_scale
        )
    )


@reward(scale=2.0)
def lin_vel_tracking(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Matching the commanded body-frame forward/lateral linear velocity."""
    return torch.exp(-scale * torch.sum((c.local_lin_vel_xy - c.commanded_lin_vel) ** 2, dim=-1))


@reward(scale=2.0)
def yaw_vel_tracking(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Matching the commanded body-frame yaw angular velocity."""
    return torch.exp(-scale * (c.local_yaw_rate - c.commands[:, CommandIndex.YAW]) ** 2)


@reward(scale=0.5)
def target_hit(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Closing the 2D distance to the commanded point-goal position."""
    return torch.exp(-scale * torch.sum((c.root_positions_xy - c.commanded_goal) ** 2, dim=-1))


@reward(scale=2.0)
def line_following(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Moving along the world-frame direction implied by the commanded body-frame velocity."""
    command = torch.cat((c.commanded_lin_vel, torch.zeros_like(c.commanded_lin_vel[:, :1])), dim=-1)
    commanded_xy = quat_apply(c.state.root_rotations, command)[:, :2]
    actual_xy = c.state.root_linear_velocities[:, :2]

    commanded_norm = torch.norm(commanded_xy, dim=-1).clamp_min(1e-6)
    actual_norm = torch.norm(actual_xy, dim=-1).clamp_min(1e-6)
    alignment = torch.sum(commanded_xy * actual_xy, dim=-1) / (commanded_norm * actual_norm)
    return torch.exp(-scale * (1.0 - alignment))


@reward(scale=1.0)
def action_rate_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Large frame-to-frame changes in action."""
    return torch.exp(-scale * torch.sum((c.actions - c.previous_actions) ** 2, dim=-1))


@reward(scale=1.0e-4)
def joint_acc(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Large joint accelerations."""
    return torch.exp(-scale * torch.sum(c.dof_accelerations**2, dim=-1))


@reward(scale=1.0e-2)
def joint_vel(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Large joint velocities."""
    return torch.exp(-scale * torch.sum(c.state.dof_velocities**2, dim=-1))


@reward(scale=1.0)
def progress(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """How much closer to the commanded goal position the last step made us. 0 on an env's first
    step and on the first step after its episode or goal changes, so a new goal earns nothing."""
    distance = torch.norm(c.commanded_goal - c.root_positions_xy, dim=-1)
    previous = c.memory.get("goal_distance", distance)
    previous = torch.where(torch.isnan(previous), distance, previous)
    c.memory["goal_distance"] = distance
    return scale * (previous - distance)


@reward(scale=4.0)
def root_lin_vel_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Root linear speed, all three axes."""
    return torch.exp(-scale * torch.sum(c.state.root_linear_velocities**2, dim=-1))


@reward(scale=2.0)
def root_ang_vel_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Root angular speed, all three axes."""
    return torch.exp(-scale * torch.sum(c.state.root_angular_velocities**2, dim=-1))


@reward(scale=10.0)
def upright(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Root tilt, as 1 minus the world-Z component of its up axis (0 upright, 1 lying flat)."""
    up_axis = torch.zeros_like(c.state.root_positions)
    up_axis[:, 2] = 1.0
    tilt = 1.0 - quat_apply(c.state.root_rotations, up_axis)[:, 2]
    return torch.exp(-scale * tilt)


@reward(scale=20.0)
def support_centered(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """XY distance from the root to the midpoint of the feet, the last two key bodies in the env's
    order (right_hand, left_hand, right_foot, left_foot): the root held over its support."""
    feet_midpoint = c.state.key_body_positions[:, -2:, :2].mean(dim=1)
    return torch.exp(-scale * torch.sum((c.root_positions_xy - feet_midpoint) ** 2, dim=-1))


@reward(scale=2.0, tolerance=0.05)
def standing_pose(c: StepContext, scale: torch.Tensor, tolerance: torch.Tensor) -> torch.Tensor:
    """Joint positions vs. the motion's `standing.pose`, ignoring up to `tolerance` rad per joint.
    Only allowed in `standing.rewards`."""
    excess = ((c.state.dof_positions - c.standing_pose).abs() - tolerance.unsqueeze(-1)).clamp_min(0.0)
    return torch.exp(-scale * torch.sum(excess**2, dim=-1))


def _heading_frame_offsets(body_positions: torch.Tensor, root_positions: torch.Tensor, root_rotations: torch.Tensor) -> torch.Tensor:
    """Bodies' (N, B, 3) offsets from the root, with XY in the root's heading frame."""
    offsets = body_positions - root_positions.unsqueeze(1)
    num_bodies = offsets.shape[1]
    rotations = root_rotations.unsqueeze(1).expand(-1, num_bodies, -1).reshape(-1, 4)
    xy = world_to_heading_frame_xy(rotations, offsets[..., :2].reshape(-1, 2)).view(-1, num_bodies, 2)
    return torch.cat((xy, offsets[..., 2:]), dim=-1)


def _heading_frame_vectors(
    vectors: torch.Tensor,
    root_rotations: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame vectors into the root's heading frame."""
    heading = yaw_quat(root_rotations)
    return quat_apply_inverse(heading, vectors)
