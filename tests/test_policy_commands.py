"""The policy's view of the commands rewrites only the goal slots: the heading rewrite never creates
or changes a velocity, behavior key or phase command. No simulator."""

import pytest
import torch

from test_termination_args import mdp  # noqa: F401  (imports the package against IsaacLab stand-ins)

from moveitmoveit.mdp import composer
from moveitmoveit.motion.manifest import COMMAND_DIM, CommandIndex

GOAL = [CommandIndex.GOAL_X, CommandIndex.GOAL_Y]
NOT_GOAL = [index for index in CommandIndex if index not in GOAL]


def _quarter_turn(rotations: torch.Tensor, vectors_xy: torch.Tensor) -> torch.Tensor:
    """A heading frame yawed by +90 degrees, in place of IsaacLab's quaternion op."""
    return torch.stack((vectors_xy[:, 1], -vectors_xy[:, 0]), dim=-1)


def _state(root_xy: torch.Tensor) -> composer.BodyState:
    n = root_xy.shape[0]
    root_positions = torch.cat((root_xy, torch.ones(n, 1)), dim=-1)
    rotations = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    zeros = torch.zeros(n, 3)
    return composer.BodyState(torch.zeros(n, 1), torch.zeros(n, 1), root_positions, rotations, zeros, zeros, torch.zeros(n, 4, 3))


def _commands(n: int) -> torch.Tensor:
    commands = torch.zeros(n, COMMAND_DIM)
    commands[:, CommandIndex.LIN_X] = 0.8
    commands[:, CommandIndex.LIN_Y] = -0.3
    commands[:, CommandIndex.YAW] = 0.5
    commands[:, CommandIndex.BINARY_KEY_0] = 1.0
    commands[:, CommandIndex.BINARY_KEY_1] = 1.0
    commands[:, CommandIndex.BINARY_KEY_2] = 1.0
    commands[:, CommandIndex.BEHAVIOR_PHASE] = 0.4
    commands[:, CommandIndex.GOAL_X] = 3.0
    commands[:, CommandIndex.GOAL_Y] = -2.0
    return commands


@pytest.mark.parametrize("local_frame", [True, False])
def test_velocity_behavior_and_phase_commands_survive(monkeypatch, local_frame):
    monkeypatch.setattr(composer, "world_to_heading_frame_xy", _quarter_turn)
    commands = _commands(3)
    original = commands.clone()
    state = _state(torch.tensor([[1.0, 1.0], [0.0, 0.0], [-2.0, 4.0]]))

    observed = composer.policy_commands(state, commands, torch.ones(3, dtype=torch.bool), local_frame)

    assert torch.equal(observed[:, NOT_GOAL], original[:, NOT_GOAL])
    offset = original[:, GOAL] - state.root_positions[:, :2]
    expected = _quarter_turn(state.root_rotations, offset) if local_frame else offset
    assert torch.equal(observed[:, GOAL], expected)
    # the env's own commands are left as they were
    assert torch.equal(commands, original)


@pytest.mark.parametrize("local_frame", [True, False])
def test_no_goal_motion_zeros_only_the_goal_slots(monkeypatch, local_frame):
    monkeypatch.setattr(composer, "world_to_heading_frame_xy", _quarter_turn)
    commands = _commands(2)
    state = _state(torch.tensor([[1.0, 1.0], [0.0, 0.0]]))
    has_goal = torch.tensor([False, True])

    observed = composer.policy_commands(state, commands, has_goal, local_frame)

    assert torch.equal(observed[:, NOT_GOAL], commands[:, NOT_GOAL])
    assert (observed[0, GOAL] == 0).all()
    assert (observed[1, GOAL] != 0).all()
