import torch

from isaaclab.utils.math import (
    euler_xyz_from_quat,
    normalize,
    quat_apply,
    quat_apply_inverse,
    quat_apply_yaw,
    quat_conjugate,
    quat_mul,
    random_yaw_orientation,
    scale_transform,
)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)

@torch.jit.script
def normalize_angle(x: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(x), torch.cos(x))

@torch.jit.script
def compute_heading_and_up(
    torso_rotation: torch.Tensor,
    inv_start_rot: torch.Tensor,
    to_target: torch.Tensor,
    vec0: torch.Tensor,
    vec1: torch.Tensor,
    up_idx: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute heading and up vectors for locomotion tasks."""
    num_envs = torso_rotation.shape[0]
    target_dirs = normalize(to_target)

    torso_quat = quat_mul(torso_rotation, inv_start_rot)
    up_vec = quat_apply(torso_quat, vec1).view(num_envs, 3)
    heading_vec = quat_apply(torso_quat, vec0).view(num_envs, 3)
    up_proj = up_vec[:, up_idx]
    heading_proj = torch.bmm(heading_vec.view(num_envs, 1, 3), target_dirs.view(num_envs, 3, 1)).view(num_envs)

    return torso_quat, up_proj, heading_proj, up_vec, heading_vec

@torch.jit.script
def compute_rot(
    torso_quat: torch.Tensor,
    velocity: torch.Tensor,
    ang_velocity: torch.Tensor,
    targets: torch.Tensor,
    torso_positions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute rotation-related quantities for locomotion tasks."""
    vel_loc = quat_apply_inverse(torso_quat, velocity)
    angvel_loc = quat_apply_inverse(torso_quat, ang_velocity)

    roll, pitch, yaw = euler_xyz_from_quat(torso_quat)

    walk_target_angle = torch.atan2(targets[:, 1] - torso_positions[:, 1], targets[:, 0] - torso_positions[:, 0])
    angle_to_target = walk_target_angle - yaw

    return vel_loc, angvel_loc, roll, pitch, yaw, angle_to_target
