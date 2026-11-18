from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import torch


@dataclass(frozen=True)
class SkeletonMap:
    dof_indexes: list[int]
    root_body_index: int
    key_body_indexes: list[int]


class BodyState(NamedTuple):
    dof_positions: torch.Tensor
    dof_velocities: torch.Tensor
    dof_accelerations: torch.Tensor
    root_positions: torch.Tensor
    root_rotations: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor
    key_body_positions: torch.Tensor


@dataclass
class StepContext:
    state: BodyState
    reference_state: BodyState
    commands: torch.Tensor
    previous_actions: torch.Tensor
    actions: torch.Tensor
    phase: torch.Tensor
    behaviors: torch.Tensor
    goal_active: torch.Tensor
    memory: dict[str, torch.Tensor] = field(default_factory=dict)
