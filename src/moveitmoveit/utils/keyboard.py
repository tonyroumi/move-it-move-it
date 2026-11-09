from __future__ import annotations

import carb
import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

from moveitmoveit.manifest import CommandIndex


_SE2_COMMANDS = (CommandIndex.LIN_X, CommandIndex.LIN_Y, CommandIndex.YAW)

# commands that aren't driven by a key binding (e.g. point-goal position); left at 0 during teleop
_NON_TELEOP_COMMANDS = (CommandIndex.GOAL_X, CommandIndex.GOAL_Y)


class Keyboard(Se2Keyboard):
    """Isaac Lab's SE(2) keyboard plus binary commands, 1.0 while held and on the first `advance`
    after a press (so a tap between two calls isn't lost): Left Shift drives binary_key_0 and
    Space binary_key_1."""

    EXTRA_KEY_MAPPING = {
        "LEFT_SHIFT": CommandIndex.BINARY_KEY_0,
        "SPACE": CommandIndex.BINARY_KEY_1,
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
    )

    def __init__(self, cfg: Se2KeyboardCfg):
        super().__init__(cfg)

        if set(self.EXTRA_KEY_MAPPING.values()) != set(self.EXTRA_COMMAND_ORDER):
            raise ValueError(
                "EXTRA_KEY_MAPPING must define exactly one key binding for "
                "every extra CommandIndex entry "
                f"(expected {set(self.EXTRA_COMMAND_ORDER)}, "
                f"got {set(self.EXTRA_KEY_MAPPING.values())})."
            )

        self._extra_commands = {
            command_name: 0.0
            for command_name in self.EXTRA_COMMAND_ORDER
        }
        # pressed since the last advance(), so a tap between two calls isn't lost
        self._pressed = set()

    def reset(self):
        """Reset both SE(2) and custom keyboard commands."""
        super().reset()

        for command_name in self._extra_commands:
            self._extra_commands[command_name] = 0.0
        self._pressed.clear()

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
        """The full (len(CommandIndex),) command, e.g. [v_x, v_y, omega_z, key_0, key_1, 0, 0];
        commands without a key binding (`_NON_TELEOP_COMMANDS`) stay 0."""
        se2_command = super().advance()

        command = torch.zeros(len(CommandIndex), dtype=se2_command.dtype, device=se2_command.device)
        command[list(_SE2_COMMANDS)] = se2_command
        for name in self.EXTRA_COMMAND_ORDER:
            command[name] = 1.0 if name in self._pressed else self._extra_commands[name]
        self._pressed.clear()

        return command

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle the standard SE(2) commands and the binary ones."""
        raw_input = event.input
        key = raw_input.name if hasattr(raw_input, "name") else None

        # Preserve all normal Se2Keyboard behavior.
        super()._on_keyboard_event(event, *args, **kwargs)

        # Ignore keys that are not custom controls (also covers CHAR events, where key is None).
        if key is None or key not in self.EXTRA_KEY_MAPPING:
            return True

        command_name = self.EXTRA_KEY_MAPPING[key]

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._extra_commands[command_name] = 1.0
            self._pressed.add(command_name)

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._extra_commands[command_name] = 0.0

        return True


class PointGoalKeyboard(Keyboard):
    """Drives the GOAL_X/GOAL_Y command: the SE(2) arrow keys move a world-frame goal offset at
    their sensitivity in m/s (integrated over `dt` per `advance`), and L resets it. The binary keys
    are as in `Keyboard`."""

    def __init__(self, cfg: Se2KeyboardCfg, dt: float, initial_offset: tuple[float, float] = (1.0, 0.0)):
        super().__init__(cfg)
        self._dt = dt
        self._initial_offset = torch.tensor(initial_offset, dtype=torch.float32, device=cfg.sim_device)
        self._goal_offset = self._initial_offset.clone()

    def reset(self):
        """Reset both the underlying SE(2) key state and the goal offset."""
        super().reset()
        self._goal_offset = self._initial_offset.clone()

    def advance(self) -> torch.Tensor:
        """The full (len(CommandIndex),) command, holding the world-frame goal offset and the binary
        keys; the caller anchors the goal to a world position."""
        command = super().advance()
        se2_command = command[list(_SE2_COMMANDS)]

        # Isaac Lab's Se2Keyboard maps left -> +v_y, right -> -v_y; negate to get left -> -y, right -> +y.
        self._goal_offset[0] += se2_command[0] * self._dt
        self._goal_offset[1] -= se2_command[1] * self._dt

        command[list(_SE2_COMMANDS)] = 0.0
        command[CommandIndex.GOAL_X] = self._goal_offset[0]
        command[CommandIndex.GOAL_Y] = self._goal_offset[1]

        return command


class NearbyGoalKeyboard(Keyboard):
    """Drives a nearby goal's held direction: the held arrow keys, in the robot's heading frame (up
    forward, down back, left to its left, right to its right), normalized so a diagonal is a unit
    vector too, or 0 when none is held (or opposite keys cancel). The direction is in GOAL_X/GOAL_Y;
    the caller places the goal from it every step. Key sensitivities don't matter, and the yaw keys
    are ignored. The binary keys are as in `Keyboard`."""

    def advance(self) -> torch.Tensor:
        """The full (len(CommandIndex),) command, holding the heading-frame goal direction and the
        binary keys, with no velocity command."""
        command = super().advance()

        # Isaac Lab's Se2Keyboard maps up -> +v_x and left -> +v_y, already the heading frame's +X/+Y
        direction = torch.sign(command[[CommandIndex.LIN_X, CommandIndex.LIN_Y]])
        norm = torch.linalg.vector_norm(direction)
        if norm > 0:
            direction = direction / norm

        command[list(_SE2_COMMANDS)] = 0.0
        command[CommandIndex.GOAL_X] = direction[0]
        command[CommandIndex.GOAL_Y] = direction[1]

        return command
