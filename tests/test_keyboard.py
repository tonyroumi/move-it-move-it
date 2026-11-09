"""play.py's keyboard commands, on stand-ins for carb and Isaac Lab's Se2Keyboard (no simulator).

Run under IsaacLab's Python, which has torch: ./isaaclab.sh -p -m pytest tests
"""

import sys
from types import SimpleNamespace

import pytest
import torch

from moveitmoveit.manifest import CommandIndex


SE2 = [CommandIndex.LIN_X, CommandIndex.LIN_Y, CommandIndex.YAW]


@pytest.fixture
def keyboards(monkeypatch):
    """`moveitmoveit.utils.keyboard`, imported against a fake Se2Keyboard whose held SE(2)
    command is set directly through `held`."""

    class Se2Keyboard:
        def __init__(self, cfg):
            self.held = torch.zeros(3)

        def reset(self):
            self.held.zero_()

        def advance(self):
            return self.held.clone()

        def _on_keyboard_event(self, event, *args, **kwargs):
            return True

    event_type = SimpleNamespace(KEY_PRESS="press", KEY_REPEAT="repeat", KEY_RELEASE="release", CHAR="char")
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(input=SimpleNamespace(KeyboardEventType=event_type)))
    monkeypatch.setitem(
        sys.modules, "isaaclab.devices", SimpleNamespace(Se2Keyboard=Se2Keyboard, Se2KeyboardCfg=object)
    )
    monkeypatch.delitem(sys.modules, "moveitmoveit.utils.keyboard", raising=False)
    from moveitmoveit.utils import keyboard

    yield keyboard
    sys.modules.pop("moveitmoveit.utils.keyboard", None)


def key_event(name: str, kind: str) -> SimpleNamespace:
    return SimpleNamespace(input=SimpleNamespace(name=name), type=kind)


def binary_keys(command: torch.Tensor) -> list[float]:
    return command[[CommandIndex.BINARY_KEY_0, CommandIndex.BINARY_KEY_1]].tolist()


def test_left_shift_is_on_while_held_and_off_after_release(keyboards):
    keyboard = keyboards.Keyboard(SimpleNamespace(sim_device="cpu"))
    assert binary_keys(keyboard.advance()) == [0.0, 0.0]

    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "press"))
    assert [binary_keys(keyboard.advance()) for _ in range(3)] == [[1.0, 0.0]] * 3

    # key repeats and the char events Kit interleaves while a key is held change nothing
    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "repeat"))
    keyboard._on_keyboard_event(SimpleNamespace(input="a", type="char"))
    assert binary_keys(keyboard.advance()) == [1.0, 0.0]

    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "release"))
    assert binary_keys(keyboard.advance()) == [0.0, 0.0]


def test_a_left_shift_tap_between_two_advances_is_not_lost(keyboards):
    keyboard = keyboards.Keyboard(SimpleNamespace(sim_device="cpu"))

    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "press"))
    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "release"))
    assert binary_keys(keyboard.advance()) == [1.0, 0.0]
    assert binary_keys(keyboard.advance()) == [0.0, 0.0]


def test_left_shift_leaves_the_other_commands_alone(keyboards):
    keyboard = keyboards.Keyboard(SimpleNamespace(sim_device="cpu"))
    keyboard.held[:] = torch.tensor([1.5, -1.0, 0.5])

    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "press"))
    keyboard._on_keyboard_event(key_event("SPACE", "press"))
    keyboard._on_keyboard_event(key_event("SPACE", "release"))
    keyboard._on_keyboard_event(key_event("RIGHT_SHIFT", "press"))
    keyboard._on_keyboard_event(key_event("UP", "press"))
    command = keyboard.advance()

    assert command[SE2].tolist() == [1.5, -1.0, 0.5]
    assert binary_keys(command) == [1.0, 1.0]  # Space was tapped since the last advance
    assert command[[CommandIndex.GOAL_X, CommandIndex.GOAL_Y]].tolist() == [0.0, 0.0]
    assert binary_keys(keyboard.advance()) == [1.0, 0.0]  # Right Shift is not bound


def test_point_goal_keyboard_also_sends_left_shift(keyboards):
    keyboard = keyboards.PointGoalKeyboard(SimpleNamespace(sim_device="cpu"), dt=0.5)
    keyboard.held[:] = torch.tensor([2.0, 0.0, 0.0])

    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "press"))
    command = keyboard.advance()

    assert command[CommandIndex.BINARY_KEY_0] == 1.0
    assert command[[CommandIndex.GOAL_X, CommandIndex.GOAL_Y]].tolist() == [2.0, 0.0]
    # the arrows only move the goal; no velocity command reaches the policy
    assert torch.count_nonzero(command[SE2]) == 0

    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "release"))
    assert keyboard.advance()[CommandIndex.BINARY_KEY_0] == 0.0
