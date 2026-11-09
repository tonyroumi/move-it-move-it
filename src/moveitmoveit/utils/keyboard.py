from __future__ import annotations

import weakref

import carb
import omni.appwindow
import torch

from isaaclab.utils.math import quat_apply_yaw

from moveitmoveit.manifest import COMMAND_DIM, CommandIndex

# key -> (forward, left, yaw) direction; each key and its numpad twin drive the same axis
MOTION_KEYS: dict[str, tuple[float, float, float]] = {
    "UP": (1.0, 0.0, 0.0),
    "NUMPAD_8": (1.0, 0.0, 0.0),
    "DOWN": (-1.0, 0.0, 0.0),
    "NUMPAD_2": (-1.0, 0.0, 0.0),
    "LEFT": (0.0, 1.0, 0.0),
    "NUMPAD_4": (0.0, 1.0, 0.0),
    "RIGHT": (0.0, -1.0, 0.0),
    "NUMPAD_6": (0.0, -1.0, 0.0),
    "Z": (0.0, 0.0, 1.0),
    "NUMPAD_7": (0.0, 0.0, 1.0),
    "X": (0.0, 0.0, -1.0),
    "NUMPAD_9": (0.0, 0.0, -1.0),
}
BINARY_KEYS: dict[str, CommandIndex] = {
    "LEFT_SHIFT": CommandIndex.BINARY_KEY_0,
    "SPACE": CommandIndex.BINARY_KEY_1,
}
RESET_KEY = "L"
MODES = ("velocity", "point_goal")


class Keyboard:
    """Drives the env's full command from the keyboard, in one of two modes.

    - ``velocity``: the arrow keys (or numpad 8/2/4/6) set the body-frame lin_x/lin_y command and
      Z/X (numpad 7/9) the yaw command, at `sensitivity` while held.
    - ``point_goal``: the goal starts `goal_offset` ahead of each env's root on the first
      `advance`, in its heading, and stays fixed in world space. The arrow keys move it at
      `sensitivity` m/s in the robot's current heading frame: up pushes it the way the robot
      faces, left/right to the robot's own left/right. Z/X do nothing.

    In both modes Left Shift drives binary_key_0 and Space binary_key_1: 1.0 while held and on the
    first `advance` after a press, so a tap between two calls isn't lost. L releases every key and,
    in ``point_goal`` mode, puts the goal back `goal_offset` ahead of the start pose.
    """

    def __init__(
        self,
        mode: str,
        dt: float,
        device: str | torch.device,
        sensitivity: tuple[float, float, float] = (1.5, 1.0, 1.0),
        goal_offset: tuple[float, float] = (1.0, 0.0),
    ):
        if mode not in MODES:
            raise ValueError(f"Unknown keyboard mode '{mode}'; expected one of {MODES}.")
        self.mode = mode
        self._dt = dt
        self._device = device
        self._sensitivity = torch.tensor(sensitivity, device=device)
        self._goal_offset = torch.tensor((*goal_offset, 0.0), device=device)

        self._held: set[str] = set()
        # binary keys pressed since the last advance()
        self._tapped: set[str] = set()
        self._start_xy: torch.Tensor | None = None
        self._start_quat: torch.Tensor | None = None
        self._goal_xy: torch.Tensor | None = None

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        # a weak proxy keeps the subscription from holding this object alive
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event)
        )

    def __del__(self):
        # __init__ may have raised before subscribing
        if getattr(self, "_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)

    def __str__(self) -> str:
        lines = [f"Keyboard ({self.mode})", f"\tReset: {RESET_KEY}"]
        if self.mode == "velocity":
            lines += [
                "\tForward / backward: Up / Down (Numpad 8 / 2)",
                "\tLeft / right: Left / Right (Numpad 4 / 6)",
                "\tYaw left / right: Z / X (Numpad 7 / 9)",
            ]
        else:
            lines += [
                "\tGoal ahead / behind: Up / Down (Numpad 8 / 2)",
                "\tGoal left / right: Left / Right (Numpad 4 / 6)",
            ]
        lines += [f"\t{command.name}: {key}" for key, command in BINARY_KEYS.items()]
        return "\n".join(lines)

    def reset(self):
        """Release every key and, in point_goal mode, put the goal back ahead of the start pose."""
        self._held.clear()
        self._tapped.clear()
        self._goal_xy = None

    def advance(self, root_xy: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        """The (N, COMMAND_DIM) command for the envs' (N, 2) root XY positions and (N, 4) root
        rotations, which only point_goal mode reads."""
        axes = torch.zeros(3, device=self._device)
        for key in self._held & MOTION_KEYS.keys():
            axes += torch.tensor(MOTION_KEYS[key], device=self._device)
        axes *= self._sensitivity

        command = torch.zeros(root_xy.shape[0], COMMAND_DIM, device=self._device)
        if self.mode == "velocity":
            command[:, CommandIndex.LIN_X : CommandIndex.YAW + 1] = axes
        else:
            if self._start_xy is None:
                self._start_xy, self._start_quat = root_xy.clone(), root_quat.clone()
            if self._goal_xy is None:
                self._goal_xy = self._start_xy + self._heading_to_world_xy(self._start_quat, self._goal_offset)
            step = torch.cat((axes[:2] * self._dt, axes.new_zeros(1)))
            self._goal_xy = self._goal_xy + self._heading_to_world_xy(root_quat, step)
            command[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1] = self._goal_xy

        for key, index in BINARY_KEYS.items():
            command[:, index] = float(key in self._held or key in self._tapped)
        self._tapped.clear()
        return command

    @staticmethod
    def _heading_to_world_xy(rotations: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
        """Rotate a (3,) heading-frame vector into world XY by each of the (N, 4) rotations' yaw."""
        return quat_apply_yaw(rotations, vector.expand(rotations.shape[0], 3))[:, :2]

    def _on_keyboard_event(self, event) -> bool:
        # CHAR events carry a character rather than a named key; only presses and releases matter
        key = getattr(event.input, "name", None)
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if key == RESET_KEY:
                self.reset()
            else:
                self._held.add(key)
                if key in BINARY_KEYS:
                    self._tapped.add(key)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._held.discard(key)
        return True
