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


@observation(dim=1)
def ref_phase(state: BodyState, local_frame: bool) -> torch.Tensor:
    """Each env's progress through the reference clip it tracks, in [0, 1]. Opt-in: no default
    selection reads it. The env fills `state.ref_phase` from `reference_phase`."""
    if state.ref_phase is None:
        raise ValueError("ref_phase is selected but the state carries no reference-clip phase.")
    return state.ref_phase.unsqueeze(-1)


def reference_phase(
    start_times: torch.Tensor, time: torch.Tensor, duration: torch.Tensor, loop: torch.Tensor
) -> torch.Tensor:
    """Progress through a clip whose clock started at `start_times` and has run for `time`, (N,).

    A looping clip wraps modulo its duration, so its phase runs [0, 1) and restarts; any other
    clip holds its last frame, so its phase clamps at 1. A zero-length clip reads as finished.
    """
    duration = duration.clamp_min(1e-6)
    clip_times = start_times + time
    clip_times = torch.where(loop, torch.remainder(clip_times, duration), clip_times)
    return (clip_times / duration).clamp(0.0, 1.0)
