from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch

from isaaclab.utils.math import quat_apply_inverse

from moveitmoveit.utils.transforms import quaternion_to_tangent_and_normal

if TYPE_CHECKING:
    from moveitmoveit.mdp.composer import BodyState


class ObservationTerm(NamedTuple):
    fn: Callable[[BodyState, bool], torch.Tensor]
    dim: int


OBSERVATIONS: dict[str, ObservationTerm] = {}


def observation(dim: int):
    def register(fn):
        OBSERVATIONS[fn.__name__] = ObservationTerm(fn, dim)
        return fn

    return register


@observation(dim=28)
def dof_pos(state: BodyState, local_frame: bool) -> torch.Tensor:
    return state.dof_positions


@observation(dim=28)
def dof_vel(state: BodyState, local_frame: bool) -> torch.Tensor:
    return state.dof_velocities


@observation(dim=1)
def root_height(state: BodyState, local_frame: bool) -> torch.Tensor:
    return state.root_positions[:, 2:3]


@observation(dim=6)
def root_orientation(state: BodyState, local_frame: bool) -> torch.Tensor:
    return quaternion_to_tangent_and_normal(state.root_rotations)


@observation(dim=3)
def root_lin_vel(state: BodyState, local_frame: bool) -> torch.Tensor:
    if local_frame:
        return quat_apply_inverse(state.root_rotations, state.root_linear_velocities)
    return state.root_linear_velocities


@observation(dim=3)
def root_ang_vel(state: BodyState, local_frame: bool) -> torch.Tensor:
    if local_frame:
        return quat_apply_inverse(state.root_rotations, state.root_angular_velocities)
    return state.root_angular_velocities


@observation(dim=12)
def key_body_offsets(state: BodyState, local_frame: bool) -> torch.Tensor:
    offsets = state.key_body_positions - state.root_positions.unsqueeze(1)
    if local_frame:
        rotations = state.root_rotations.unsqueeze(1).expand(-1, offsets.shape[1], -1)
        offsets = quat_apply_inverse(rotations, offsets)
    return offsets.flatten(1)
