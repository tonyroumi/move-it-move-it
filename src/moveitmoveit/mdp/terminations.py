from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, NamedTuple

import torch

if TYPE_CHECKING:
    from moveitmoveit.mdp.composer import StepContext


class TerminationTerm(NamedTuple):
    fn: Callable[..., torch.Tensor]
    defaults: dict[str, float]


TERMINATIONS: dict[str, TerminationTerm] = {}


def termination(**defaults: float):
    def register(fn):
        TERMINATIONS[fn.__name__] = TerminationTerm(fn, defaults)
        return fn

    return register


@termination(threshold=1.5)
def deviation_from_motion(c: StepContext, threshold: float) -> torch.Tensor:
    """The tracked pose deviates too far (summed squared DOF error) from the reference motion."""
    return torch.sum((c.state.dof_positions - c.targets.dof_positions) ** 2, dim=-1) > threshold


@termination(height_threshold=0.52)
def unhealthy(c: StepContext, height_threshold: float) -> torch.Tensor:
    """The root body falls below a minimum height."""
    return c.state.root_positions[:, 2] < height_threshold
