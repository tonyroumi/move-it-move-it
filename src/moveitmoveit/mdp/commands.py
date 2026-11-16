from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum
from typing import TYPE_CHECKING, NamedTuple

import torch

from moveitmoveit.utils.transforms import world_to_heading_frame_xy

if TYPE_CHECKING:
    from moveitmoveit.state import StepContext


class CommandIndex(IntEnum):
    LIN_X = 0
    LIN_Y = 1
    YAW = 2
    BINARY_KEY_0 = 3
    BINARY_KEY_1 = 4
    GOAL_X = 5
    GOAL_Y = 6
    BINARY_KEY_2 = 7
    BEHAVIOR_PHASE = 8


COMMAND_DIM = len(CommandIndex)


class CommandTerm(NamedTuple):
    fn: Callable[[StepContext, bool], torch.Tensor]
    indexes: tuple[CommandIndex, ...]


COMMANDS: dict[str, CommandTerm] = {}


def command(*indexes: CommandIndex):
    """Register conversion of a command's slots to policy inputs."""
    def register(fn):
        COMMANDS[fn.__name__] = CommandTerm(fn, indexes)
        return fn

    return register


@command(CommandIndex.LIN_X, CommandIndex.LIN_Y)
def linear_velocity(c: StepContext, local_frame: bool) -> torch.Tensor:
    """Velocity commands are already expressed in their command frame."""
    return c.commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1]


@command(CommandIndex.YAW)
def yaw_rate(c: StepContext, local_frame: bool) -> torch.Tensor:
    return c.commands[:, CommandIndex.YAW : CommandIndex.YAW + 1]


@command(CommandIndex.GOAL_X, CommandIndex.GOAL_Y)
def point_goal(c: StepContext, local_frame: bool) -> torch.Tensor:
    """Observe a world target relative to the root, optionally in its heading frame."""
    goal = c.commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1]
    offset = goal - c.state.root_positions[:, :2]
    if local_frame:
        offset = world_to_heading_frame_xy(c.state.root_rotations, offset)
    return torch.where(c.goal_active.unsqueeze(-1), offset, torch.zeros_like(offset))


def command_observation(c: StepContext, local_frame: bool) -> torch.Tensor:
    """Convert independent commands without changing storage or unregistered slots."""
    observation = c.commands.clone()
    for term in COMMANDS.values():
        observation[:, term.indexes] = term.fn(c, local_frame)
    return observation
