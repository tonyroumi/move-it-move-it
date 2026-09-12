from __future__ import annotations

import carb
import torch

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

from moveitmoveit.commands import CommandIndex

# CommandIndex entries handled by the base Se2Keyboard controller (v_x, v_y,
# omega_z), and therefore excluded from the dynamically derived extra/binary
# command set below.
_SE2_COMMANDS = (CommandIndex.LIN_X, CommandIndex.LIN_Y, CommandIndex.YAW)


class Keyboard(Se2Keyboard):
    """
    Extends Isaac Lab's SE(2) keyboard controller with additional binary
    command inputs.

    The set and order of extra binary commands is derived from
    `CommandIndex`: any entry not handled by the base Se2Keyboard (v_x, v_y,
    omega_z) becomes an extra binary command, ordered by its `CommandIndex`
    value. Adding/removing a `CommandIndex` entry requires adding/removing
    its corresponding key binding in `EXTRA_KEY_MAPPING`.

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

    # ---------------------------------------------------------------------
    # Explicit custom key mappings
    # ---------------------------------------------------------------------

    EXTRA_KEY_MAPPING = {
        "LEFT_SHIFT": CommandIndex.BINARY_KEY_0,
        "SPACE": CommandIndex.BINARY_KEY_1,
    }

    # Derived from CommandIndex rather than hand-maintained, so the extra
    # command set/order stays in sync with commands.py automatically.
    EXTRA_COMMAND_ORDER = tuple(
        sorted(
            (command for command in CommandIndex if command not in _SE2_COMMANDS),
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

    def advance(self) -> torch.Tensor:
        """
        Return the complete keyboard command.

        Returns:
            Tensor with shape (3 + num_extra_commands,).

            Example:
                [v_x, v_y, omega_z, binary_key_0, binary_key_1]
        """
        se2_command = super().advance()

        extra_command = torch.tensor(
            [
                self._extra_commands[name]
                for name in self.EXTRA_COMMAND_ORDER
            ],
            dtype=se2_command.dtype,
            device=se2_command.device,
        )

        return torch.cat((se2_command, extra_command))

    def _on_keyboard_event(self, event, *args, **kwargs):
        """
        Handle standard SE(2) commands and additional binary commands.
        """

        # Preserve all normal Se2Keyboard behavior.
        super()._on_keyboard_event(event, *args, **kwargs)

        key = event.input.name

        # Ignore keys that are not custom controls.
        if key not in self.EXTRA_KEY_MAPPING:
            return True

        command_name = self.EXTRA_KEY_MAPPING[key]

        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._extra_commands[command_name] = 1.0

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._extra_commands[command_name] = 0.0

        return True
