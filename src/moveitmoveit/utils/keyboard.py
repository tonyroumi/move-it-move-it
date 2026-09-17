from __future__ import annotations

import carb
import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

from moveitmoveit.commands import CommandIndex


_SE2_COMMANDS = (CommandIndex.LIN_X, CommandIndex.LIN_Y, CommandIndex.YAW)

# commands that aren't driven by a key binding (e.g. point-goal position); left at 0 during teleop
_NON_TELEOP_COMMANDS = (CommandIndex.GOAL_X, CommandIndex.GOAL_Y)


class Keyboard(Se2Keyboard):
    """
    Extends Isaac Lab's SE(2) keyboard controller with additional binary
    command inputs.

    Output:
        [v_x, v_y, omega_z, *extra_commands]

    Standard SE(2) keys:
        Forward:        Up Arrow / Numpad 8
        Backward:       Down Arrow / Numpad 2
        Left/Right:     Arrow keys / Numpad 4, 6
        Yaw:            Z / X or Numpad 7, 9

    Additional binary keys:
        Left Shift:     binary_key_0
        Space:          binary_key_1

    Binary controls are:
        0.0 when released
        1.0 while held
    """

    EXTRA_KEY_MAPPING = {
        "LEFT_SHIFT": CommandIndex.BINARY_KEY_0,
        "SPACE": CommandIndex.BINARY_KEY_1,
    }

    # Derived from CommandIndex rather than hand-maintained, so the extra
    # command set/order stays in sync with commands.py automatically.
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

    def reset(self):
        """Reset both SE(2) and custom keyboard commands."""
        super().reset()

        for command_name in self._extra_commands:
            self._extra_commands[command_name] = 0.0

    def __str__(self) -> str:
        msg = super().__str__()
        msg += "\n\t----------------------------------------------\n"
        msg += "\n".join(
            f"\t{command_name.name}: {key}"
            for key, command_name in self.EXTRA_KEY_MAPPING.items()
        )
        return msg

    def advance(self) -> torch.Tensor:
        """
        Return the complete keyboard command.

        Returns:
            Tensor with shape (len(CommandIndex),), indexed by CommandIndex. Commands not driven
            by a key binding (see `_NON_TELEOP_COMMANDS`) are left at 0.

            Example:
                [v_x, v_y, omega_z, binary_key_0, binary_key_1, 0.0, 0.0]
        """
        se2_command = super().advance()

        command = torch.zeros(len(CommandIndex), dtype=se2_command.dtype, device=se2_command.device)
        command[list(_SE2_COMMANDS)] = se2_command
        for name in self.EXTRA_COMMAND_ORDER:
            command[name] = self._extra_commands[name]

        return command

    def _on_keyboard_event(self, event, *args, **kwargs):
        """
        Handle standard SE(2) commands and additional binary commands.
        """

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

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._extra_commands[command_name] = 0.0

        return True


class PointGoalKeyboard(Se2Keyboard):
    """
    Drives the point-goal command (GOAL_X, GOAL_Y) from the keyboard instead of a velocity command.

    Reuses Isaac Lab's standard SE(2) arrow/numpad bindings, but integrates them into a persistent
    2D offset that grows while a key is held, rather than emitting an instantaneous velocity -- i.e.
    a point that walks away from its starting position as you steer it. The offset is in world-frame
    meters, not rotated by the robot's heading (matching how goal commands are sampled in
    `MotionManager`/`MotionLearningEnv`); the caller is responsible for anchoring it to the robot's
    current position before writing it into the env's `commands` buffer.

    Keys:
        Forward/Backward:  Up Arrow / Numpad 8, Down Arrow / Numpad 2  -> +X / -X
        Left/Right:        Arrow keys / Numpad 4, 6                    -> -Y / +Y
        Reset:             L                                            -> back to `initial_offset`

    Yaw keys (Z / X, Numpad 7 / 9) have no effect in this mode.
    """

    def __init__(self, cfg: Se2KeyboardCfg, initial_offset: tuple[float, float] = (1.0, 0.0)):
        super().__init__(cfg)
        self._initial_offset = torch.tensor(initial_offset, dtype=torch.float32, device=cfg.sim_device)
        self._goal_offset = self._initial_offset.clone()

    def reset(self):
        """Reset both the underlying SE(2) key state and the goal offset."""
        super().reset()
        self._goal_offset = self._initial_offset.clone()

    def advance(self) -> torch.Tensor:
        """
        Return the complete keyboard command.

        Returns:
            Tensor with shape (len(CommandIndex),), indexed by CommandIndex. Only GOAL_X/GOAL_Y are
            non-zero, holding the accumulated world-frame offset (not yet anchored to the robot's
            position).
        """
        se2_command = super().advance()

        # Isaac Lab's Se2Keyboard maps left -> +v_y, right -> -v_y; negate to get left -> -y, right -> +y.
        self._goal_offset[0] += se2_command[0]
        self._goal_offset[1] -= se2_command[1]

        command = torch.zeros(len(CommandIndex), dtype=se2_command.dtype, device=se2_command.device)
        command[CommandIndex.GOAL_X] = self._goal_offset[0]
        command[CommandIndex.GOAL_Y] = self._goal_offset[1]

        return command
