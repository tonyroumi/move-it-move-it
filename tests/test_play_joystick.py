"""play.py's --joystick commands are relative to the robot's heading, at any yaw and while turning.

'velocity' mode: the keyboard's lin/yaw command is already body-frame, so play.py passes it through
and the policy and rewards read it against the body-frame velocity (no second rotation).
'point_goal' mode: each step's keyboard offset change is rotated by the robot's current yaw before
it moves the world-space goal. Without --joystick the env's sampled commands are untouched.
"""

import importlib.util
import math
import sys
import types

import pytest
import torch

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from conftest import REPO_ROOT  # noqa: E402
from moveitmoveit import mdp  # noqa: E402
from moveitmoveit.manifest import CommandIndex  # noqa: E402
from moveitmoveit.mdp import BodyState, StepContext, policy_commands  # noqa: E402
from moveitmoveit.utils import transforms  # noqa: E402

GOAL = slice(CommandIndex.GOAL_X, CommandIndex.GOAL_Y + 1)
VELOCITY = slice(CommandIndex.LIN_X, CommandIndex.YAW + 1)
HEADINGS = (0.0, 90.0, 180.0, 270.0)


def yaw_quat(degrees: float, pitch_degrees: float = 0.0) -> torch.Tensor:
    """(1, 4) xyzw rotation: yaw about Z after pitch about Y."""
    yaw, pitch = math.radians(degrees), math.radians(pitch_degrees)
    q_yaw = torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])
    q_pitch = torch.tensor([[0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)]])
    return transforms.quat_mul(q_yaw, q_pitch)


def heading(degrees: float) -> torch.Tensor:
    """(2,) world-space unit vector the robot faces at `degrees` of yaw."""
    return torch.tensor([math.cos(math.radians(degrees)), math.sin(math.radians(degrees))])


def left_of(degrees: float) -> torch.Tensor:
    return heading(degrees + 90.0)


@pytest.fixture(scope="module")
def play():
    """scripts/play.py imported with its launcher/CLI stubbed, so no simulator starts."""
    cli = types.ModuleType("isaaclab_tasks.utils")
    cli.add_launcher_args = lambda parser: None
    cli.launch_simulation = cli.resolve_task_config = None
    cli.setup_preset_cli = lambda parser: (parser.parse_args(["--checkpoint", "unused.pt"]), [])

    with pytest.MonkeyPatch.context() as patch:
        patch.setitem(sys.modules, "isaaclab_tasks", types.ModuleType("isaaclab_tasks"))
        patch.setitem(sys.modules, "isaaclab_tasks.utils", cli)
        patch.setitem(sys.modules, "isaaclab_tasks_experimental", types.ModuleType("isaaclab_tasks_experimental"))
        patch.setattr(sys, "argv", ["play.py"])
        spec = importlib.util.spec_from_file_location("play_under_test", REPO_ROOT / "scripts/play.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


@pytest.fixture
def point_goal_keyboard(monkeypatch):
    """The real PointGoalKeyboard on a stand-in Se2Keyboard whose held keys are set directly."""

    class Se2Keyboard:
        def __init__(self, cfg):
            self.held = torch.zeros(3)
            self.callbacks = {}

        def reset(self):
            self.held.zero_()

        def add_callback(self, key, func):
            self.callbacks[key] = func

        def press_l(self):
            # Se2Keyboard on L: reset itself, then run the additional callback
            self.reset()
            self.callbacks["L"]()

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


def keyboard_offset(keyboard, num_envs: int = 1) -> torch.Tensor:
    """The keyboard's goal offset as play.py hands it on: expanded to (num_envs, 2)."""
    return keyboard.advance().unsqueeze(0).expand(num_envs, -1)[:, GOAL].clone()


# --- point_goal mode --------------------------------------------------------------------------


@pytest.mark.parametrize("yaw", HEADINGS)
def test_point_goal_starts_ahead_and_up_pushes_it_the_way_the_robot_faces(play, point_goal_keyboard, yaw):
    keyboard = point_goal_keyboard
    anchor = torch.tensor([[3.0, -2.0]])
    rotation = yaw_quat(yaw)
    goal = play.HeadingRelativeGoal(anchor, rotation)

    start = goal.update(keyboard_offset(keyboard), rotation)[0]
    assert start.tolist() == pytest.approx((anchor[0] + heading(yaw)).tolist(), abs=1e-5)

    keyboard.held[:] = torch.tensor([2.0, 0.0, 0.0])  # up arrow, 2 m/s for dt=0.5 s
    moved = goal.update(keyboard_offset(keyboard), rotation)[0]
    assert (moved - start).tolist() == pytest.approx(heading(yaw).tolist(), abs=1e-5)

    keyboard.held[:] = torch.tensor([-2.0, 0.0, 0.0])  # down arrow
    back = goal.update(keyboard_offset(keyboard), rotation)[0]
    assert back.tolist() == pytest.approx(start.tolist(), abs=1e-5)

    keyboard.held[:] = torch.tensor([0.0, 2.0, 0.0])  # left arrow
    side = goal.update(keyboard_offset(keyboard), rotation)[0]
    assert (side - back).tolist() == pytest.approx(left_of(yaw).tolist(), abs=1e-5)


@pytest.mark.parametrize("yaw", HEADINGS)
def test_left_and_right_arrows_move_the_point_goal_to_the_robots_own_left_and_right(play, point_goal_keyboard, yaw):
    keyboard = point_goal_keyboard
    rotation = yaw_quat(yaw)
    goal = play.HeadingRelativeGoal(torch.zeros(1, 2), rotation)
    start = goal.update(keyboard_offset(keyboard), rotation)[0].clone()

    # Se2Keyboard: left arrow -> +v_y, right arrow -> -v_y
    keyboard.held[:] = torch.tensor([0.0, 2.0, 0.0])
    left = goal.update(keyboard_offset(keyboard), rotation)[0].clone()
    assert (left - start).tolist() == pytest.approx(left_of(yaw).tolist(), abs=1e-5)

    keyboard.held[:] = torch.tensor([0.0, -2.0, 0.0])
    goal.update(keyboard_offset(keyboard), rotation)
    right = goal.update(keyboard_offset(keyboard), rotation)[0]
    assert (right - left).tolist() == pytest.approx((-2.0 * left_of(yaw)).tolist(), abs=1e-5)


def test_left_arrow_follows_the_robots_left_while_turning(play, point_goal_keyboard):
    keyboard = point_goal_keyboard
    goal = play.HeadingRelativeGoal(torch.zeros(1, 2), yaw_quat(0.0))
    expected = goal.update(keyboard_offset(keyboard), yaw_quat(0.0))[0].clone()

    keyboard.held[:] = torch.tensor([0.0, 0.2, 0.0])  # left arrow while turning 0 -> 180 degrees
    for yaw in torch.linspace(20.0, 180.0, 9).tolist():
        expected += 0.1 * left_of(yaw)
        actual = goal.update(keyboard_offset(keyboard), yaw_quat(yaw))[0]
        assert actual.tolist() == pytest.approx(expected.tolist(), abs=1e-5)


def test_point_goal_follows_the_live_heading_while_turning(play, point_goal_keyboard):
    keyboard = point_goal_keyboard
    goal = play.HeadingRelativeGoal(torch.zeros(1, 2), yaw_quat(0.0))
    expected = goal.update(keyboard_offset(keyboard), yaw_quat(0.0))[0].clone()

    # hold up while the robot turns left 0 -> 90 degrees: each step pushes along that step's heading
    keyboard.held[:] = torch.tensor([0.2, 0.0, 0.0])
    for yaw in torch.linspace(10.0, 90.0, 9).tolist():
        expected += 0.1 * heading(yaw)
        actual = goal.update(keyboard_offset(keyboard), yaw_quat(yaw))[0]
        assert actual.tolist() == pytest.approx(expected.tolist(), abs=1e-5)

    # facing +Y now; earlier steps are not re-rotated, and up keeps pushing along +Y
    keyboard.held[:] = torch.tensor([2.0, 0.0, 0.0])
    actual = goal.update(keyboard_offset(keyboard), yaw_quat(90.0))[0]
    assert actual.tolist() == pytest.approx((expected + torch.tensor([0.0, 1.0])).tolist(), abs=1e-5)


def test_point_goal_stays_fixed_in_world_space_while_no_key_is_held(play, point_goal_keyboard):
    keyboard = point_goal_keyboard
    goal = play.HeadingRelativeGoal(torch.tensor([[1.0, 1.0]]), yaw_quat(0.0))
    start = goal.update(keyboard_offset(keyboard), yaw_quat(0.0)).clone()

    for yaw in (45.0, 90.0, 180.0, 270.0):
        assert torch.equal(goal.update(keyboard_offset(keyboard), yaw_quat(yaw)), start)


def test_point_goal_uses_yaw_only(play, point_goal_keyboard):
    keyboard = point_goal_keyboard
    goal = play.HeadingRelativeGoal(torch.zeros(1, 2), yaw_quat(90.0, pitch_degrees=30.0))
    start = goal.update(keyboard_offset(keyboard), yaw_quat(90.0, pitch_degrees=30.0))[0]
    assert start.tolist() == pytest.approx([0.0, 1.0], abs=1e-5)  # a full 1 m ahead, not cos(30) of it

    keyboard.held[:] = torch.tensor([2.0, 0.0, 0.0])
    moved = goal.update(keyboard_offset(keyboard), yaw_quat(90.0, pitch_degrees=-20.0))[0]
    assert (moved - start).tolist() == pytest.approx([0.0, 1.0], abs=1e-5)


def test_point_goal_moves_each_env_in_its_own_heading(play, point_goal_keyboard):
    keyboard = point_goal_keyboard
    rotations = torch.cat([yaw_quat(yaw) for yaw in HEADINGS])
    anchors = torch.arange(8, dtype=torch.float32).view(4, 2)
    goal = play.HeadingRelativeGoal(anchors, rotations)
    start = goal.update(keyboard_offset(keyboard, 4), rotations).clone()

    keyboard.held[:] = torch.tensor([2.0, 0.0, 0.0])
    moved = goal.update(keyboard_offset(keyboard, 4), rotations)
    expected = torch.stack([heading(yaw) for yaw in HEADINGS])
    assert (moved - start).flatten().tolist() == pytest.approx(expected.flatten().tolist(), abs=1e-5)


def test_l_resets_the_point_goal_to_its_start(play, point_goal_keyboard):
    keyboard = point_goal_keyboard
    anchor = torch.tensor([[2.0, 0.0]])
    goal = play.HeadingRelativeGoal(anchor, yaw_quat(90.0))
    keyboard.add_callback("L", goal.reset)  # as play.py registers it
    start = goal.update(keyboard_offset(keyboard), yaw_quat(90.0)).clone()

    keyboard.held[:] = torch.tensor([2.0, 2.0, 0.0])
    for yaw in (120.0, 200.0, 300.0):
        goal.update(keyboard_offset(keyboard), yaw_quat(yaw))

    keyboard.press_l()
    reset = goal.update(keyboard_offset(keyboard), yaw_quat(300.0))
    assert reset[0].tolist() == pytest.approx(start[0].tolist(), abs=1e-5)


# --- velocity mode and the non-joystick path --------------------------------------------------


def body_state(rotation: torch.Tensor, linear_velocity: torch.Tensor, angular_velocity: torch.Tensor) -> BodyState:
    return BodyState(
        torch.zeros(1, 28),
        torch.zeros(1, 28),
        torch.tensor([[3.0, -2.0, 0.9]]),
        rotation,
        linear_velocity,
        angular_velocity,
        torch.zeros(1, 4, 3),
    )


def velocity_command(lin_x: float, lin_y: float, yaw_rate: float) -> torch.Tensor:
    command = torch.zeros(1, len(CommandIndex))
    command[0, VELOCITY] = torch.tensor([lin_x, lin_y, yaw_rate])
    return command


@pytest.mark.parametrize("turn_rate", (0.0, 1.0), ids=("steady", "turning"))
@pytest.mark.parametrize("yaw", HEADINGS)
def test_velocity_command_is_body_frame_with_no_second_rotation(yaw, turn_rate):
    # up + left arrows with a yaw rate: what Keyboard.advance() hands play.py, which writes it as-is
    command = velocity_command(1.5, 1.0, turn_rate)
    rotation = yaw_quat(yaw)
    world_velocity = 1.5 * heading(yaw) + 1.0 * left_of(yaw)
    state = body_state(
        rotation,
        torch.cat((world_velocity, torch.zeros(1))).unsqueeze(0),
        torch.tensor([[0.0, 0.0, turn_rate]]),
    )

    for local_frame in (True, False):
        observed = policy_commands(state, command, torch.tensor([False]), local_frame)
        assert torch.equal(observed[:, VELOCITY], command[:, VELOCITY])

    # moving forward-left along the robot's own heading at the commanded yaw rate tracks perfectly
    context = StepContext(state, torch.zeros(1, 28), command, torch.zeros(1, 28), torch.zeros(1, 28), None)
    assert mdp.lin_vel_tracking(context, torch.tensor(2.0)).item() == pytest.approx(1.0, abs=1e-5)
    assert mdp.yaw_vel_tracking(context, torch.tensor(2.0)).item() == pytest.approx(1.0, abs=1e-5)

    # moving along world +X instead is only "forward" when the robot faces +X
    state = body_state(rotation, torch.tensor([[1.5, 0.0, 0.0]]), torch.tensor([[0.0, 0.0, turn_rate]]))
    command = velocity_command(1.5, 0.0, turn_rate)
    context = StepContext(state, torch.zeros(1, 28), command, torch.zeros(1, 28), torch.zeros(1, 28), None)
    assert (mdp.lin_vel_tracking(context, torch.tensor(2.0)).item() == pytest.approx(1.0, abs=1e-5)) == (yaw == 0.0)

