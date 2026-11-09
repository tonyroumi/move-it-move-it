from __future__ import annotations

import weakref

import carb
import omni.appwindow
import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

from moveitmoveit.manifest import CommandIndex


_SE2_COMMANDS = (CommandIndex.LIN_X, CommandIndex.LIN_Y, CommandIndex.YAW)

# commands that aren't driven by a key binding (e.g. point-goal position, or the behavior phase the
# env fills in); left at 0 during teleop
_NON_TELEOP_COMMANDS = (CommandIndex.GOAL_X, CommandIndex.GOAL_Y, CommandIndex.BEHAVIOR_PHASE)


class Keyboard(Se2Keyboard):
    """Isaac Lab's SE(2) keyboard plus binary commands, 1.0 while held and on the first `advance`
    after a press (so a tap between two calls isn't lost): Left Shift drives binary_key_0, Space
    binary_key_1 and C binary_key_2."""

    EXTRA_KEY_MAPPING = {
        "LEFT_SHIFT": CommandIndex.BINARY_KEY_0,
        "SPACE": CommandIndex.BINARY_KEY_1,
        "C": CommandIndex.BINARY_KEY_2,
    }

    # derived from CommandIndex, so the extra command set/order stays in sync with it
    EXTRA_COMMAND_ORDER = tuple(
        sorted(
            (
                command
                for command in CommandIndex
                if command not in _SE2_COMMANDS and command not in _NON_TELEOP_COMMANDS
            ),
            key=int,
        )

    def __del__(self):
        # __init__ may have raised before subscribing
        if getattr(self, "_sub", None) is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)

    def __str__(self) -> str:
        """The SE(2) bindings followed by the binary ones."""
        msg = super().__str__()
        msg += "\n\t----------------------------------------------\n"
        msg += "\n".join(
            f"\t{command_name.name}: {key}"
            for key, command_name in self.EXTRA_KEY_MAPPING.items()
        )
        return msg

    def advance(self) -> torch.Tensor:
        """The full (len(CommandIndex),) command, e.g. [v_x, v_y, omega_z, key_0, key_1, 0, 0, key_2, 0];
        commands without a key binding (`_NON_TELEOP_COMMANDS`) stay 0."""
        se2_command = super().advance()

        command = torch.zeros(len(CommandIndex), dtype=se2_command.dtype, device=se2_command.device)
        command[list(_SE2_COMMANDS)] = se2_command
        for name in self.EXTRA_COMMAND_ORDER:
            command[name] = 1.0 if name in self._pressed else self._extra_commands[name]
        self._pressed.clear()

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
