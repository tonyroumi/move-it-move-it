"""A faint on-screen keyboard overlay that highlights the keys bound by ``isaaclab.devices.Se2Keyboard``.

Subscribes to raw keyboard events independently of the control device, so it only
visualizes which of the bound keys are currently held -- it does not drive the command
itself.
"""

from __future__ import annotations

import carb
import omni
import omni.appwindow
import omni.ui as ui

# groups mirror isaaclab.devices.Se2Keyboard._create_key_bindings: each command axis is
# driven by either its numpad or arrow-key binding, so both keys light up the same cell
_KEY_GROUPS: dict[str, tuple[str, ...]] = {
    "fwd": ("NUMPAD_8", "UP"),
    "back": ("NUMPAD_2", "DOWN"),
    "left": ("NUMPAD_4", "LEFT"),
    "right": ("NUMPAD_6", "RIGHT"),
    "yaw_pos": ("NUMPAD_7", "Z"),
    "yaw_neg": ("NUMPAD_9", "X"),
    "reset": ("L",),
}

_LABELS: dict[str, str] = {
    "fwd": "FWD",
    "back": "BACK",
    "left": "LEFT",
    "right": "RIGHT",
    "yaw_pos": "YAW+",
    "yaw_neg": "YAW-",
    "reset": "RESET",
}

# 3x3 grid resembling a keyboard's arrow-key cluster plus a yaw/reset row below it
_LAYOUT: list[list[str | None]] = [
    [None, "fwd", None],
    ["left", "back", "right"],
    ["yaw_pos", "reset", "yaw_neg"],
]

_IDLE_COLOR = 0x22FFFFFF  # faint translucent white (AABBGGRR)
_ACTIVE_COLOR = 0xFF50E635  # solid green
_LABEL_COLOR = 0xCCFFFFFF


class KeyboardOverlay:
    """Floating, mostly-transparent window that lights up the currently held joystick keys."""

    def __init__(self):
        self._pressed: set[str] = set()
        self._key_rects: dict[str, ui.Rectangle] = {}

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(self._keyboard, self._on_keyboard_event)

        self._window = ui.Window(
            "Joystick Keys",
            width=160,
            height=160,
            position_x=20,
            position_y=20,
            flags=(
                ui.WINDOW_FLAGS_NO_TITLE_BAR
                | ui.WINDOW_FLAGS_NO_RESIZE
                | ui.WINDOW_FLAGS_NO_SCROLLBAR
                | ui.WINDOW_FLAGS_NO_MOVE
                | ui.WINDOW_FLAGS_NO_BACKGROUND
            ),
        )
        self._build_ui()

    def close(self):
        """Unsubscribe from keyboard events. Call when done with the overlay."""
        if self._sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
            self._sub = None

    def __del__(self):
        self.close()

    def _on_keyboard_event(self, event, *args, **kwargs):
        name = event.input.name
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._pressed.add(name)
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._pressed.discard(name)
        return True

    def _build_ui(self):
        with self._window.frame:
            with ui.VStack(spacing=4, style={"margin": 4}):
                for row in _LAYOUT:
                    with ui.HStack(spacing=4, height=44):
                        for group in row:
                            if group is None:
                                ui.Spacer(width=44)
                                continue
                            with ui.ZStack(width=44):
                                rect = ui.Rectangle(style={"background_color": _IDLE_COLOR, "border_radius": 4})
                                ui.Label(
                                    _LABELS[group],
                                    alignment=ui.Alignment.CENTER,
                                    style={"font_size": 11, "color": _LABEL_COLOR},
                                )
                                self._key_rects[group] = rect

    def update(self):
        """Refresh key-cap colors from the current pressed-key state. Call once per frame."""
        for group, rect in self._key_rects.items():
            active = any(key in self._pressed for key in _KEY_GROUPS[group])
            rect.set_style({"background_color": _ACTIVE_COLOR if active else _IDLE_COLOR, "border_radius": 4})
