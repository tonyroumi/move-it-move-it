from enum import IntEnum

import torch

from .utils import transforms


class RewardIndex(IntEnum):
    MOTION_TRACKING = 0
    LIN_VEL_TRACKING = 1
    YAW_VEL_TRACKING = 2
    TARGET_HIT = 3


NUM_REWARDS = len(RewardIndex)


@torch.jit.script
def motion_tracking_reward(
    dof_positions: torch.Tensor,
    ref_dof_positions: torch.Tensor,
    key_body_positions: torch.Tensor,
    root_positions: torch.Tensor,
    ref_key_body_positions: torch.Tensor,
    ref_root_positions: torch.Tensor,
    dof_scale: float = 2.0,
    key_body_scale: float = 5.0,
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
    scale: float = 4.0,
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
    scale: float = 4.0,
) -> torch.Tensor:
    """Reward for matching the commanded body-frame yaw angular velocity."""
    local_ang_vel = transforms.quat_apply_inverse(root_rotations, root_angular_velocities)  # (N, 3)
    error = (local_ang_vel[:, 2] - commanded_yaw_vel) ** 2  # (N,)
    return torch.exp(-scale * error)


def target_hit_reward(num_envs: int, device: torch.device) -> torch.Tensor:
    """Placeholder: no target-position state exists yet for the binary key commands, so this is always zero."""
    return torch.zeros(num_envs, device=device)
