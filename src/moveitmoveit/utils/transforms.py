import torch

from isaaclab.utils.math import quat_apply, quat_mul


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
def world_to_heading_frame_xy(
    rotations: torch.Tensor,
    vectors_xy: torch.Tensor,
) -> torch.Tensor:
    """Rotate world-frame XY vectors into the body's yaw-only heading frame."""
    x_axis = torch.zeros_like(rotations[..., :3])
    x_axis[..., 0] = 1.0

    heading = quat_apply(rotations, x_axis)
    yaw = torch.atan2(heading[..., 1], heading[..., 0])

    cos, sin = torch.cos(yaw), torch.sin(yaw)
    x, y = vectors_xy.unbind(dim=-1)

    return torch.stack(
        (cos * x + sin * y, -sin * x + cos * y),
        dim=-1,
    )


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

    root_position = body_positions[:, ref_body_index]

    # Express body positions as world-frame offsets from the root.
    body_offsets = body_positions - root_position.unsqueeze(1)


    quat = quat.unsqueeze(1).expand(-1, num_bodies, 4)
    # Rotate the offsets, then translate back to the root.
    body_positions = root_position.unsqueeze(1) + quat_apply(quat, body_offsets)

    # Rotate world-frame orientations and velocities.
    body_rotations = quat_mul(quat, body_rotations)
    body_linear_velocities = quat_apply(quat, body_linear_velocities)
    body_angular_velocities = quat_apply(quat, body_angular_velocities)

    return body_positions, body_rotations, body_linear_velocities, body_angular_velocities


def heading_frame(vectors: torch.Tensor, root_rotations: torch.Tensor) -> torch.Tensor:
    """Express world-frame vectors in the root's heading frame. """
    rotations = root_rotations.reshape(
        root_rotations.shape[0], *([1] * (vectors.ndim - 2)), 4
    )

    xy = world_to_heading_frame_xy(rotations, vectors[..., :2])
    return torch.cat((xy, vectors[..., 2:]), dim=-1)