"""Termination terms: when an env's episode ends early."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .composer import StepContext


TERMINATIONS: dict[str, Callable[[StepContext], torch.Tensor]] = {}


def termination(fn):
    TERMINATIONS[fn.__name__] = fn
    return fn


@termination
def deviation_from_motion(c: StepContext, threshold: float = 1.5) -> torch.Tensor:
    """The tracked pose deviates too far (summed squared DOF error) from the reference motion."""
    return torch.sum((c.state.dof_positions - c.targets.dof_positions) ** 2, dim=-1) > threshold


@termination
def unhealthy(c: StepContext, height_threshold: float = 0.52) -> torch.Tensor:
    """The root body falls below a minimum height."""
    return c.state.root_positions[:, 2] < height_threshold
