from enum import IntEnum

import torch


class TerminationIndex(IntEnum):
    DEVIATION_FROM_MOTION = 0
    UNHEALTHY = 1


NUM_TERMINATIONS = len(TerminationIndex)


@torch.jit.script
def deviation_from_motion_termination(
    dof_positions: torch.Tensor,  # (N, num_dofs)
    ref_dof_positions: torch.Tensor,  # (N, num_dofs)
    threshold: float = 1.0,
) -> torch.Tensor:
    """Terminates when the tracked pose deviates too far (summed squared DOF error) from the reference motion."""
    deviation = torch.sum((dof_positions - ref_dof_positions) ** 2, dim=-1)  # (N,)
    return deviation > threshold


@torch.jit.script
def unhealthy_termination(
    root_height: torch.Tensor,
    height_threshold: float = 0.65,
) -> torch.Tensor:
    """Terminates when the root body falls below a minimum height."""
    return root_height < height_threshold
