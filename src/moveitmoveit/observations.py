import torch

from moveitmoveit.utils import transforms


@torch.jit.script
def compute_proprioceptive_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    body_positions: torch.Tensor,
    local_frame: bool = False,
) -> torch.Tensor:
    num_envs = root_positions.shape[0]
    num_bodies = body_positions.shape[1]

    # Body positions relative to the root.
    body_offsets = body_positions - root_positions.unsqueeze(1)

    if local_frame:
        # Express root velocities in the root's local frame.
        lin_vel = transforms.quat_apply_inverse(
            root_rotations,
            root_linear_velocities,
        )
        ang_vel = transforms.quat_apply_inverse(
            root_rotations,
            root_angular_velocities,
        )

        # Apply the inverse root rotation to every body offset.
        root_rotations_expanded = root_rotations.unsqueeze(1).expand(
            num_envs, num_bodies, 4
        )
        body_positions_local = transforms.quat_apply_inverse(
            root_rotations_expanded,
            body_offsets,
        )

        body_positions_flat = body_positions_local.reshape(num_envs, -1)

    else:
        lin_vel = root_linear_velocities
        ang_vel = root_angular_velocities
        body_positions_flat = body_offsets.reshape(num_envs, -1)

    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            transforms.quaternion_to_tangent_and_normal(root_rotations),
            lin_vel,
            ang_vel,
            body_positions_flat,
        ),
        dim=-1,
    )
