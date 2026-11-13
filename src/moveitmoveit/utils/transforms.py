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
    yaw_quat
)


@torch.jit.script
def quaternion_to_tangent_and_normal(q: torch.Tensor) -> torch.Tensor:
    """The rotated x (tangent) and z (normal) axes, concatenated: (..., 6)."""
    ref_tangent = torch.zeros_like(q[..., :3])
    ref_normal = torch.zeros_like(q[..., :3])
    ref_tangent[..., 0] = 1
    ref_normal[..., -1] = 1
    tangent = quat_apply(q, ref_tangent)
    normal = quat_apply(q, ref_normal)
    return torch.cat([tangent, normal], dim=len(tangent.shape) - 1)


@torch.jit.script
def world_to_heading_frame_xy(rotations: torch.Tensor, vectors_xy: torch.Tensor) -> torch.Tensor:
    """Rotate (N, 2) world-frame XY vectors into each body's yaw-only heading frame, so a tilted
    body still sees a vector ahead of it along +X and one to its left along +Y."""
    x_axis = torch.zeros_like(rotations[:, :3])
    x_axis[:, 0] = 1.0
    heading = quat_apply(rotations, x_axis)
    yaw = torch.atan2(heading[:, 1], heading[:, 0])
    cos, sin = torch.cos(yaw), torch.sin(yaw)
    x, y = vectors_xy[:, 0], vectors_xy[:, 1]
    return torch.stack((cos * x + sin * y, -sin * x + cos * y), dim=-1)


@torch.jit.script
def apply_random_yaw(
    quat: torch.Tensor,
    ref_body_index: int,
    body_positions: torch.Tensor,
    body_rotations: torch.Tensor,
    body_linear_velocities: torch.Tensor,
    body_angular_velocities: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply a yaw rotation to a sampled reference motion about its root."""
    num_bodies = body_positions.shape[1]

    # Root position is the pivot of the rotation.
    root_position = body_positions[:, ref_body_index]

    # Express body positions as world-frame offsets from the root.
    body_offsets = body_positions - root_position.unsqueeze(1)

    # Expand yaw rotation across all bodies.
    quat = quat.unsqueeze(1).expand(-1, num_bodies, 4)

    # Rotate the offsets, then translate back to the root.
    # The resulting positions are still world-frame positions.
    body_positions = root_position.unsqueeze(1) + quat_apply(quat, body_offsets)

    # Rotate world-frame orientations and velocities.
    body_rotations = quat_mul(quat, body_rotations)
    body_linear_velocities = quat_apply(quat, body_linear_velocities)
    body_angular_velocities = quat_apply(quat, body_angular_velocities)

    return body_positions, body_rotations, body_linear_velocities, body_angular_velocities
