from enum import IntEnum

import torch

from .utils import transforms


class RewardIndex(IntEnum):
    MOTION_TRACKING = 0
    LIN_VEL_TRACKING = 1
    YAW_VEL_TRACKING = 2
    TARGET_HIT = 3
    LINE_FOLLOWING = 4
    ACTION_RATE_L2 = 5
    JOINT_ACC = 6
    JOINT_VEL = 7


NUM_REWARDS = len(RewardIndex)


REWARD_KWARG_DEFAULTS: dict[str, dict[str, float]] = {
    "motion_tracking": {"dof_scale": 2.0, "key_body_scale": 5.0},
    "lin_vel_tracking": {"scale": 2.0},
    "yaw_vel_tracking": {"scale": 2.0},
    "target_hit": {"scale": 0.5},
    "line_following": {"scale": 2.0},
    "action_rate_l2": {"scale": 1.0},
    "joint_acc": {"scale": 1.0e-4},
    "joint_vel": {"scale": 1.0e-2},
}


@torch.jit.script
def motion_tracking_reward(
    dof_positions: torch.Tensor,
    ref_dof_positions: torch.Tensor,
    key_body_positions: torch.Tensor,
    root_positions: torch.Tensor,
    ref_key_body_positions: torch.Tensor,
    ref_root_positions: torch.Tensor,
    dof_scale: torch.Tensor,
    key_body_scale: torch.Tensor,
) -> torch.Tensor:
    """Pose-tracking reward: DOF positions and root-relative key-body offsets vs. the reference motion."""
    dof_error = torch.sum((dof_positions - ref_dof_positions) ** 2, dim=-1)

    key_body_offsets = key_body_positions - root_positions.unsqueeze(1)
    ref_key_body_offsets = ref_key_body_positions - ref_root_positions.unsqueeze(1)
    num_envs = key_body_offsets.shape[0]
    key_body_error = torch.sum(
        (key_body_offsets - ref_key_body_offsets).reshape(num_envs, -1) ** 2, dim=-1
    )

    return torch.exp(-dof_scale * dof_error - key_body_scale * key_body_error)


@torch.jit.script
def lin_vel_tracking_reward(
    root_linear_velocities: torch.Tensor,
    root_rotations: torch.Tensor,
    commanded_lin_vel: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reward for matching the commanded body-frame forward/lateral linear velocity."""
    local_lin_vel = transforms.quat_apply_inverse(root_rotations, root_linear_velocities)[:, :2]
    error = torch.sum((local_lin_vel - commanded_lin_vel) ** 2, dim=-1)  # (N,)
    return torch.exp(-scale * error)


@torch.jit.script
def yaw_vel_tracking_reward(
    root_angular_velocities: torch.Tensor,
    root_rotations: torch.Tensor,
    commanded_yaw_vel: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reward for matching the commanded body-frame yaw angular velocity."""
    local_ang_vel = transforms.quat_apply_inverse(root_rotations, root_angular_velocities)  # (N, 3)
    error = (local_ang_vel[:, 2] - commanded_yaw_vel) ** 2  # (N,)
    return torch.exp(-scale * error)


@torch.jit.script
def target_hit_reward(
    root_positions: torch.Tensor,
    commanded_goal_positions: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reward for closing the 2D distance to the commanded point-goal position."""
    error = torch.sum((root_positions - commanded_goal_positions) ** 2, dim=-1)
    return torch.exp(-scale * error)


@torch.jit.script
def line_following_reward(
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    commanded_lin_vel: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Reward for moving along the line running through the torso in the world-frame direction implied
    by the commanded body-frame linear velocity."""
    commanded_dir_local = torch.cat(
        (commanded_lin_vel, torch.zeros_like(commanded_lin_vel[:, :1])), dim=-1
    )
    commanded_dir_world = transforms.quat_apply(root_rotations, commanded_dir_local)

    commanded_dir_xy = commanded_dir_world[:, :2]
    actual_vel_xy = root_linear_velocities[:, :2]

    commanded_norm = torch.norm(commanded_dir_xy, dim=-1).clamp_min(1e-6)
    actual_norm = torch.norm(actual_vel_xy, dim=-1).clamp_min(1e-6)

    # cosine similarity
    alignment = torch.sum(commanded_dir_xy * actual_vel_xy, dim=-1) / (commanded_norm * actual_norm)
    error = 1.0 - alignment
    return torch.exp(-scale * error)


@torch.jit.script
def action_rate_l2_reward(
    actions: torch.Tensor,
    previous_actions: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Penalize large frame-to-frame changes in action, for smooth/continuous motion."""
    error = torch.sum((actions - previous_actions) ** 2, dim=-1)
    return torch.exp(-scale * error)


@torch.jit.script
def joint_acc_reward(
    joint_accelerations: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Penalize large joint accelerations, for smooth/natural motion."""
    error = torch.sum(joint_accelerations ** 2, dim=-1)
    return torch.exp(-scale * error)


@torch.jit.script
def joint_vel_reward(
    joint_velocities: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Penalize large joint velocities, for smooth/natural motion."""
    error = torch.sum(joint_velocities ** 2, dim=-1)
    return torch.exp(-scale * error)
    