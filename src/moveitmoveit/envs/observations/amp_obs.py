

import torch

from moveitmoveit.utils import transforms


@torch.jit.script
def compute_amp_observations(
    dof_positions: torch.Tensor,  # (N, num_dofs)
    dof_velocities: torch.Tensor,  # (N, num_dofs)
    root_positions: torch.Tensor,  # (N, 3)
    root_rotations: torch.Tensor,  # (N, 4), xyzw
    root_linear_velocities: torch.Tensor,  # (N, 3), world frame
    root_angular_velocities: torch.Tensor,  # (N, 3), world frame
    key_body_positions: torch.Tensor,  # (N, num_key_bodies, 3)
) -> torch.Tensor:
    num_envs = root_positions.shape[0]

    # base-frame velocities: world-frame quantities rotated into the robot's own frame
    local_lin_vel = transforms.quat_apply_inverse(root_rotations, root_linear_velocities)  # (N, 3)
    local_ang_vel = transforms.quat_apply_inverse(root_rotations, root_angular_velocities)  # (N, 3)

    # key-body offsets expressed in the base frame
    key_offsets_w = key_body_positions - root_positions.unsqueeze(1)  # (N, num_key_bodies, 3)
    num_key_bodies = key_offsets_w.shape[1]
    key_rotations = root_rotations.unsqueeze(1).expand(num_envs, num_key_bodies, 4)
    local_key_positions = transforms.quat_apply_inverse(key_rotations, key_offsets_w).reshape(num_envs, -1)  # (N, num_key_bodies*3)

    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            transforms.quaternion_to_tangent_and_normal(root_rotations),
            local_lin_vel,
            local_ang_vel,
            local_key_positions,
        ),
        dim=-1,
    )
