
from __future__ import annotations

import weakref

import carb
import omni.appwindow
import torch

from moveitmoveit.mdp.commands import COMMAND_DIM, CommandIndex


MOTION_KEYS = {
    "UP": (1, 0, 0),
    "NUMPAD_8": (1, 0, 0),
    "DOWN": (-1, 0, 0),
    "NUMPAD_2": (-1, 0, 0),
    "LEFT": (0, 1, 0),
    "NUMPAD_4": (0, 1, 0),
    "RIGHT": (0, -1, 0),
    "NUMPAD_6": (0, -1, 0),
    "Z": (0, 0, 1),
    "NUMPAD_7": (0, 0, 1),
    "X": (0, 0, -1),
    "NUMPAD_9": (0, 0, -1),
}

BINARY_KEYS = {
    "LEFT_SHIFT": CommandIndex.BINARY_KEY_0,
    "RIGHT_SHIFT": CommandIndex.BINARY_KEY_1,
    "C": CommandIndex.BINARY_KEY_2,
}

RESET_KEY = "L"


class Keyboard:
    """Keyboard-driven velocity, point-goal, and binary commands."""

    def __init__(
        self,
        mode: str,
        dt: float,
        device: str | torch.device,
        sensitivity: tuple[float, float, float] = (1.5, 1.0, 1.0),
        goal_offset: tuple[float, float] = (1.0, 0.0),
    ):
        if mode not in ("velocity", "point_goal"):
            raise ValueError(f"Unknown keyboard mode: {mode}")

        self.mode = mode
        self._dt = dt
        self._device = device
        self._sensitivity = torch.tensor(sensitivity, device=device)
        self._goal_offset = torch.tensor(goal_offset, device=device)

        self._motion_keys = {
            key: torch.tensor(direction, device=device) * self._sensitivity
            for key, direction in MOTION_KEYS.items()
        }

        self._held: set[str] = set()
        self._tapped: set[str] = set()
        self._goal_xy: torch.Tensor | None = None

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event),
        )

    def __del__(self):
        if getattr(self, "_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)

    def __str__(self) -> str:
        lines = [
            f"Keyboard ({self.mode})",
            f"  Reset: {RESET_KEY}",
            "  Forward / backward: Up / Down (Numpad 8 / 2)",
            "  Left / right: Left / Right (Numpad 4 / 6)",
        ]

        if self.mode == "velocity":
            lines.append("  Yaw left / right: Z / X (Numpad 7 / 9)")

        lines.extend(f"  {index.name}: {key}" for key, index in BINARY_KEYS.items())
        return "\n".join(lines)

    def reset(self) -> None:
        """Release all keys and reset the point goal."""
        self._held.clear()
        self._tapped.clear()
        self._goal_xy = None

    def advance(self, root_xy: torch.Tensor, root_quat: torch.Tensor) -> torch.Tensor:
        """Return keyboard commands for each environment."""
        num_envs = root_xy.shape[0]
        commands = torch.zeros(num_envs, COMMAND_DIM, device=self._device)

        axes = torch.zeros(3, device=self._device)
        for key in self._held:
            if key in self._motion_keys:
                axes += self._motion_keys[key]

        if self.mode == "velocity":
            commands[:, CommandIndex.LIN_X : CommandIndex.YAW + 1] = axes
        else:
            if self._goal_xy is None:
                self._goal_xy = root_xy.clone() + self._goal_offset

            self._goal_xy += axes[:2] * self._dt
            commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1] = self._goal_xy

        for key, index in BINARY_KEYS.items():
            commands[:, index] = float(key in self._held or key in self._tapped)

        self._tapped.clear()
        return commands

    def _on_keyboard_event(self, event) -> bool:
        """Track key presses, releases, and resets."""
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
