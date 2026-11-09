"""Stand-ins for the parts of carb and Isaac Lab the tests touch, so they run on CPU without a simulator.

Run under IsaacLab's Python, which has torch: ./isaaclab.sh -p -m pytest tests
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


# --- isaaclab.utils.math ------------------------------------------------------------------------
# Used only when Isaac Lab isn't importable. xyzw quaternions, as Isaac Lab's; TorchScript-typed,
# because moveitmoveit.utils.transforms scripts functions that call these.


def _quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., :3]
    t = 2.0 * torch.linalg.cross(xyz, vec, dim=-1)
    return vec + quat[..., 3:] * t + torch.linalg.cross(xyz, t, dim=-1)


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat((-quat[..., :3], quat[..., 3:]), dim=-1)


def _quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return _quat_apply(_quat_conjugate(quat), vec)


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return torch.stack(
        (
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ),
        dim=-1,
    )


def _normalize(x: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    return x / x.norm(p=2.0, dim=-1).clamp(min=eps).unsqueeze(-1)


def _euler_xyz_from_quat(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("not used by the tests")


def _unused(*args, **kwargs):
    raise NotImplementedError("not used by the tests")


def _isaaclab_math_stub() -> ModuleType:
    module = ModuleType("isaaclab.utils.math")
    module.quat_apply = _quat_apply
    module.quat_apply_inverse = _quat_apply_inverse
    module.quat_conjugate = _quat_conjugate
    module.quat_mul = _quat_mul
    module.normalize = _normalize
    module.euler_xyz_from_quat = _euler_xyz_from_quat
    module.quat_apply_yaw = _unused
    module.random_yaw_orientation = _unused
    module.scale_transform = _unused
    return module


try:
    import isaaclab.utils.math  # noqa: F401
except ImportError:
    for name in ("isaaclab", "isaaclab.utils"):
        sys.modules.setdefault(name, ModuleType(name))
    sys.modules["isaaclab.utils.math"] = _isaaclab_math_stub()


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion product, for building test rotations."""
    return _quat_mul(q1, q2)


# --- carb and Isaac Lab's Se2Keyboard -----------------------------------------------------------


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
