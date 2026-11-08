"""World-space goal tracking: sampling, the policy's view of the goal, progress and the keyboard."""

import math
import sys
import types

import pytest
import torch

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from conftest import REPO_ROOT  # noqa: E402
from moveitmoveit.manifest import CommandIndex, load_manifest  # noqa: E402
from moveitmoveit.mdp import BodyState, ComposedMDP, StepContext, policy_commands  # noqa: E402
from moveitmoveit.motion_set import MotionSet  # noqa: E402
from moveitmoveit.utils import transforms  # noqa: E402

GOAL = slice(CommandIndex.GOAL_X, CommandIndex.GOAL_Y + 1)
OBS = ("root_height",)


def yaw_quat(yaw: float, pitch: float = 0.0) -> torch.Tensor:
    """xyzw rotation: yaw about Z after pitch about Y."""
    q_yaw = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
    q_pitch = torch.tensor([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)])
    return transforms.quat_mul(q_yaw.unsqueeze(0), q_pitch.unsqueeze(0))[0]


def body_state(root_positions: torch.Tensor, root_rotations: torch.Tensor) -> BodyState:
    n = root_positions.shape[0]
    return BodyState(
        torch.zeros(n, 28),
        torch.zeros(n, 28),
        root_positions,
        root_rotations,
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n, 4, 3),
    )


def goal_commands(goals_xy: torch.Tensor) -> torch.Tensor:
    commands = torch.zeros(goals_xy.shape[0], len(CommandIndex))
    commands[:, GOAL] = goals_xy
    return commands


def step_context(root_xy: torch.Tensor, commands: torch.Tensor) -> StepContext:
    n = root_xy.shape[0]
    roots = torch.cat((root_xy, torch.ones(n, 1)), dim=-1)
    rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(n, 4)

    def no_reference():
        raise AssertionError("goal terms must not sample the reference motion")

    return StepContext(
        body_state(roots, rotations), torch.zeros(n, 28), commands, torch.zeros(n, 28), torch.zeros(n, 28), no_reference
    )


# --- coordinate transforms --------------------------------------------------------------------


def test_heading_frame_puts_a_goal_ahead_on_plus_x_and_to_the_left_on_plus_y():
    facing_plus_y = yaw_quat(math.pi / 2).expand(2, 4)
    world = torch.tensor([[0.0, 2.0], [-3.0, 0.0]])  # ahead of it, and to its left

    local = transforms.world_to_heading_frame_xy(facing_plus_y, world)

    assert local.tolist() == [pytest.approx([2.0, 0.0], abs=1e-6), pytest.approx([0.0, 3.0], abs=1e-6)]


def test_heading_frame_ignores_pitch():
    tilted = yaw_quat(math.pi / 4, pitch=0.6).unsqueeze(0)
    upright = yaw_quat(math.pi / 4).unsqueeze(0)
    world = torch.tensor([[1.0, 2.0]])

    assert torch.allclose(
        transforms.world_to_heading_frame_xy(tilted, world), transforms.world_to_heading_frame_xy(upright, world), atol=1e-6
    )


def test_policy_sees_the_goal_relative_to_its_root():
    rotations = torch.stack((yaw_quat(0.0), yaw_quat(math.pi)))
    roots = torch.tensor([[10.0, 20.0, 1.0], [-5.0, 0.0, 1.0]])
    commands = goal_commands(torch.tensor([[11.0, 20.0], [-6.0, 0.0]]))  # both 1 m along -/+ world X
    has_goal = torch.tensor([True, True])

    local = policy_commands(body_state(roots, rotations), commands, has_goal, local_frame=True)
    world = policy_commands(body_state(roots, rotations), commands, has_goal, local_frame=False)

    assert local[:, GOAL].tolist() == [pytest.approx([1.0, 0.0], abs=1e-6)] * 2  # each 1 m straight ahead
    assert world[:, GOAL].tolist() == [pytest.approx([1.0, 0.0]), pytest.approx([-1.0, 0.0])]
    assert torch.equal(local[:, : CommandIndex.GOAL_X], commands[:, : CommandIndex.GOAL_X])
    assert torch.equal(commands[:, GOAL], torch.tensor([[11.0, 20.0], [-6.0, 0.0]]))  # the world goal is untouched


def test_envs_without_a_goal_observe_zero_there():
    roots = torch.tensor([[3.0, 4.0, 1.0], [3.0, 4.0, 1.0]])
    rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(2, 4)
    commands = goal_commands(torch.zeros(2, 2))

    observed = policy_commands(body_state(roots, rotations), commands, torch.tensor([True, False]), local_frame=True)

    assert observed[0, GOAL].tolist() == pytest.approx([-3.0, -4.0])
    assert observed[1, GOAL].tolist() == [0.0, 0.0]


# --- resampling -------------------------------------------------------------------------------


@pytest.fixture
def motions(write_manifest, write_clip) -> MotionSet:
    return MotionSet.from_manifest(
        write_manifest(
            {
                "to_goal": {"file": write_clip("a", 5), "commands": {"goal_x": [1.0, 2.0], "goal_y": [-1.0, -0.5]}},
                "velocity": {"file": write_clip("b", 6), "commands": {"lin_x": [0.5, 1.0]}},
            }
        ),
        "cpu",
    )


def test_goals_are_sampled_per_env_around_each_root_in_world_space(motions):
    torch.manual_seed(0)
    motion_ids = torch.tensor([0, 0, 0, 1])
    roots_xy = torch.tensor([[0.0, 0.0], [100.0, 0.0], [0.0, -50.0], [7.0, 7.0]])

    commands = motions.sample_commands(motion_ids, roots_xy)

    offsets = commands[:3, GOAL] - roots_xy[:3]
    assert ((offsets[:, 0] >= 1.0) & (offsets[:, 0] <= 2.0)).all()
    assert ((offsets[:, 1] >= -1.0) & (offsets[:, 1] <= -0.5)).all()
    assert len({tuple(row) for row in offsets.tolist()}) == 3  # each env draws its own goal
    assert commands[3, GOAL].tolist() == [0.0, 0.0]  # a motion without a goal isn't anchored
    assert motions.has_goal.tolist() == [True, False]


def test_a_zeroed_goal_command_puts_the_goal_at_the_root(write_manifest, write_clip):
    motions = MotionSet.from_manifest(
        write_manifest({"m": {"file": write_clip("a", 5), "commands": {"goal_x": [1.0, 2.0]}, "zero_command_prob": 1.0}}),
        "cpu",
    )
    roots_xy = torch.tensor([[3.0, -4.0]])

    assert torch.equal(motions.sample_commands(torch.tensor([0]), roots_xy)[:, GOAL], roots_xy)


def test_sampling_without_roots_is_unchanged(motions):
    torch.manual_seed(0)
    anchored = motions.sample_commands(torch.tensor([0, 1]), torch.zeros(2, 2))
    torch.manual_seed(0)
    assert torch.equal(motions.sample_commands(torch.tensor([0, 1])), anchored)


def test_shipped_goal_manifest_declares_goal_rewards():
    specs = load_manifest(str(REPO_ROOT / "src/configs/goal_manifest.yaml"))
    mdp = ComposedMDP(specs, OBS, OBS, "cpu")

    assert {"goal_x", "goal_y"} <= set(specs[0].commands)
    assert {"progress", "target_hit"} <= set(mdp.reward_names)


# --- reward direction and reset semantics -----------------------------------------------------


@pytest.fixture
def progress_mdp(write_manifest) -> ComposedMDP:
    specs = load_manifest(write_manifest({"m": {"file": "m.npz", "rewards": {"progress": 1.0, "target_hit": 1.0}}}))
    return ComposedMDP(specs, OBS, OBS, "cpu")


def test_progress_rewards_walking_toward_the_goal_and_penalizes_walking_away(progress_mdp):
    motion_ids = torch.zeros(2, dtype=torch.long)
    commands = goal_commands(torch.tensor([[5.0, 0.0], [5.0, 0.0]]))

    progress_mdp.rewards(step_context(torch.tensor([[0.0, 0.0], [0.0, 0.0]]), commands), motion_ids)
    _, raw, _ = progress_mdp.rewards(step_context(torch.tensor([[0.5, 0.0], [-0.5, 0.0]]), commands), motion_ids)

    assert raw["progress"].tolist() == pytest.approx([0.5, -0.5])
    assert raw["target_hit"][0] > raw["target_hit"][1]  # closer is better


def test_progress_starts_over_for_reset_envs_only(progress_mdp):
    motion_ids = torch.zeros(2, dtype=torch.long)
    commands = goal_commands(torch.tensor([[5.0, 0.0], [5.0, 0.0]]))
    progress_mdp.rewards(step_context(torch.tensor([[0.0, 0.0], [0.0, 0.0]]), commands), motion_ids)

    # env 1 gets a new goal far away; without the reset its distance jump would read as -10
    progress_mdp.reset_memory(torch.tensor([1]))
    commands[1, GOAL] = torch.tensor([-5.0, 0.0])
    _, raw, _ = progress_mdp.rewards(step_context(torch.tensor([[1.0, 0.0], [0.0, 0.0]]), commands), motion_ids)
    assert raw["progress"].tolist() == pytest.approx([1.0, 0.0])

    _, raw, _ = progress_mdp.rewards(step_context(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]), commands), motion_ids)
    assert raw["progress"].tolist() == pytest.approx([0.0, 1.0])


# --- keyboard ---------------------------------------------------------------------------------


@pytest.fixture
def point_goal_keyboard(monkeypatch):
    """PointGoalKeyboard on a stand-in for Isaac Lab's Se2Keyboard, whose held keys are set directly."""

    class Se2Keyboard:
        def __init__(self, cfg):
            self._sim_device = cfg.sim_device
            self.held = torch.zeros(3)

        def reset(self):
            self.held.zero_()

        def advance(self):
            return self.held.clone()

    monkeypatch.setitem(sys.modules, "carb", types.ModuleType("carb"))
    monkeypatch.setitem(
        sys.modules, "isaaclab.devices", types.SimpleNamespace(Se2Keyboard=Se2Keyboard, Se2KeyboardCfg=object)
    )
    monkeypatch.delitem(sys.modules, "moveitmoveit.utils.keyboard", raising=False)
    from moveitmoveit.utils.keyboard import PointGoalKeyboard

    yield PointGoalKeyboard(types.SimpleNamespace(sim_device="cpu"), dt=0.5)
    sys.modules.pop("moveitmoveit.utils.keyboard", None)


def test_keyboard_moves_the_goal_only_while_a_key_is_held(point_goal_keyboard):
    keyboard = point_goal_keyboard
    assert keyboard.advance()[GOAL].tolist() == [1.0, 0.0]

    keyboard.held[:] = torch.tensor([2.0, 0.0, 0.0])  # up arrow, 2 m/s
    keyboard.advance()
    assert keyboard.advance()[GOAL].tolist() == [3.0, 0.0]

    keyboard.held[:] = torch.tensor([0.0, 1.0, 0.0])  # left arrow: Se2Keyboard's +v_y, the goal's -Y
    assert keyboard.advance()[GOAL].tolist() == [3.0, -0.5]

    keyboard.held.zero_()
    command = keyboard.advance()
    assert command[GOAL].tolist() == [3.0, -0.5]  # released keys leave the goal where it is
    assert torch.count_nonzero(command[: CommandIndex.GOAL_X]) == 0

    keyboard.reset()
    assert keyboard.advance()[GOAL].tolist() == [1.0, 0.0]
