"""Point-goal keyboard mode moves the goal in fixed world XY axes, whatever way the robot faces. No simulator."""

import importlib
import math
import sys
from types import SimpleNamespace

import pytest
import torch

from moveitmoveit.motion.manifest import CommandIndex

DT = 0.1
GOAL = slice(CommandIndex.GOAL_X, CommandIndex.GOAL_Y + 1)


@pytest.fixture
def keyboard_module(monkeypatch):
    """moveitmoveit.utils.keyboard, imported against stand-ins for the Kit modules that need Isaac Sim."""
    press, release = object(), object()
    keyboard = SimpleNamespace()
    carb = SimpleNamespace(
        input=SimpleNamespace(
            KeyboardEventType=SimpleNamespace(KEY_PRESS=press, KEY_RELEASE=release),
            acquire_input_interface=lambda: SimpleNamespace(
                subscribe_to_keyboard_events=lambda *args: object(),
                unsubscribe_to_keyboard_events=lambda *args: None,
            ),
        )
    )
    appwindow = SimpleNamespace(get_default_app_window=lambda: SimpleNamespace(get_keyboard=lambda: keyboard))
    monkeypatch.setitem(sys.modules, "carb", carb)
    monkeypatch.setitem(sys.modules, "omni", SimpleNamespace(appwindow=appwindow))
    monkeypatch.setitem(sys.modules, "omni.appwindow", appwindow)
    monkeypatch.delitem(sys.modules, "moveitmoveit.utils.keyboard", raising=False)
    module = importlib.import_module("moveitmoveit.utils.keyboard")
    yield module
    sys.modules.pop("moveitmoveit.utils.keyboard", None)


def _yaw_quat(yaw: float, n: int) -> torch.Tensor:
    """(n, 4) wxyz rotations of `yaw` radians about +Z."""
    return torch.tensor([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]).repeat(n, 1)


def _press(module, keyboard, name: str) -> None:
    event = SimpleNamespace(type=module.carb.input.KeyboardEventType.KEY_PRESS, input=SimpleNamespace(name=name))
    keyboard._on_keyboard_event(event)


def _goal_after_holding(module, key: str, yaw: float, steps: int = 5) -> torch.Tensor:
    """The (2, 2) goal after holding `key` for `steps` advances on two envs facing `yaw`."""
    keyboard = module.Keyboard("point_goal", dt=DT, device="cpu", sensitivity=(1.0, 1.0, 1.0), goal_offset=(0.0, 0.0))
    root_xy = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    quat = _yaw_quat(yaw, 2)
    keyboard.advance(root_xy, quat)
    _press(module, keyboard, key)
    for _ in range(steps):
        command = keyboard.advance(root_xy, quat)
    return command[:, GOAL] - root_xy


@pytest.mark.parametrize("yaw", [0.0, math.pi / 2, math.pi, -0.7, 2.3])
@pytest.mark.parametrize(
    "key, world",
    [("UP", (1.0, 0.0)), ("DOWN", (-1.0, 0.0)), ("LEFT", (0.0, 1.0)), ("RIGHT", (0.0, -1.0)), ("NUMPAD_8", (1.0, 0.0))],
)
def test_arrow_keys_move_the_goal_along_fixed_world_axes(keyboard_module, key, world, yaw):
    moved = _goal_after_holding(keyboard_module, key, yaw)
    expected = torch.tensor(world) * DT * 5
    assert torch.allclose(moved, expected.expand(2, 2), atol=1e-6)


def test_start_offset_is_world_aligned(keyboard_module):
    keyboard = keyboard_module.Keyboard("point_goal", dt=DT, device="cpu", goal_offset=(1.0, 0.5))
    root_xy = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    command = keyboard.advance(root_xy, _yaw_quat(math.pi / 2, 2))
    assert torch.allclose(command[:, GOAL], root_xy + torch.tensor([1.0, 0.5]))


def test_goal_ignores_the_robot_turning_while_keys_are_held(keyboard_module):
    keyboard = keyboard_module.Keyboard("point_goal", dt=DT, device="cpu", sensitivity=(1.0, 1.0, 1.0), goal_offset=(0.0, 0.0))
    root_xy = torch.zeros(1, 2)
    keyboard.advance(root_xy, _yaw_quat(0.0, 1))
    _press(keyboard_module, keyboard, "UP")
    for yaw in (0.0, 1.0, 2.0, 3.0):
        command = keyboard.advance(root_xy, _yaw_quat(yaw, 1))
    assert torch.allclose(command[:, GOAL], torch.tensor([[4 * DT, 0.0]]), atol=1e-6)


def test_velocity_mode_is_unchanged(keyboard_module):
    keyboard = keyboard_module.Keyboard("velocity", dt=DT, device="cpu", sensitivity=(2.0, 1.0, 1.0))
    _press(keyboard_module, keyboard, "UP")
    command = keyboard.advance(torch.zeros(2, 2), _yaw_quat(1.0, 2))
    assert torch.allclose(command[:, CommandIndex.LIN_X], torch.full((2,), 2.0))
    assert torch.count_nonzero(command[:, GOAL]) == 0


def test_help_text_describes_world_axes(keyboard_module):
    text = str(keyboard_module.Keyboard("point_goal", dt=DT, device="cpu"))
    assert "world" in text and "ahead" not in text and "behind" not in text
