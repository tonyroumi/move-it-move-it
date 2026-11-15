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
    """Terminate if joint positions deviate too far from the target motion."""
    return torch.sum((c.state.dof_positions - c.reference_state.dof_positions) ** 2, dim=-1) > threshold


@termination(height_threshold=0.55)
def unhealthy(c: StepContext, height_threshold: float) -> torch.Tensor:
    """Terminate if the root height falls below a minimum threshold."""
    return c.state.root_positions[:, 2] < height_threshold
