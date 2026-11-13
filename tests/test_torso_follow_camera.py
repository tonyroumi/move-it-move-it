"""The torso-follow camera starts where the "none" camera stays and translates with the torso. No simulator."""

import sys
from types import SimpleNamespace

import pytest
import torch

from test_episode_reset_behaviors import MotionLearningEnv  # imported against IsaacLab stand-ins

TORSO = 2
START_EYE = (7.5, 7.5, 7.5)
START_LOOKAT = (0.0, 0.0, 0.0)
TORSO_START = torch.tensor([10.0, -20.0, 0.9])


@pytest.fixture
def views(monkeypatch) -> list[tuple[tuple, tuple]]:
    """Every (eye, target) the Kit viewport is sent, in place of the renderer module that needs Isaac Sim."""
    sent = []
    kit = SimpleNamespace(set_kit_renderer_camera_view=lambda eye, target: sent.append((eye, target)))
    monkeypatch.setitem(sys.modules, "isaaclab_physx.renderers.kit_viewport_utils", kit)
    return sent


def _env() -> MotionLearningEnv:
    """A MotionLearningEnv with the state its torso-follow camera reads, as `_setup_env` leaves it."""
    env = MotionLearningEnv.__new__(MotionLearningEnv)
    env.cfg = SimpleNamespace(camera_type="torso-follow", viewer=SimpleNamespace(eye=START_EYE, lookat=START_LOOKAT))
    env.ref_body_index = TORSO
    # two envs of four bodies: only env 0's torso may move the camera
    body_pos = torch.full((2, 4, 3), 123.0)
    body_pos[0, TORSO] = TORSO_START
    env.robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=SimpleNamespace(torch=body_pos)))
    env.sim = SimpleNamespace(set_camera_view=lambda eye, target: None)

    env._camera_start_eye = torch.tensor(env.cfg.viewer.eye, dtype=torch.float)
    env._camera_start_target = torch.tensor(env.cfg.viewer.lookat, dtype=torch.float)
    env._camera_torso_anchor = None
    env._camera_smoothing_alpha = 0.1
    env._camera_eye_smooth = None
    env._camera_target_smooth = None
    return env


def _move_torso(env: MotionLearningEnv, displacement: torch.Tensor) -> None:
    env.robot.data.body_pos_w.torch[0, TORSO] = TORSO_START + displacement


def _view(views: list[tuple[tuple, tuple]]) -> tuple[torch.Tensor, torch.Tensor]:
    eye, target = views[-1]
    return torch.tensor(eye), torch.tensor(target)


def test_camera_starts_at_the_viewer_vantage_wherever_the_torso_is(views):
    env = _env()

    env._update_camera()

    assert views == [(START_EYE, START_LOOKAT)]


def test_camera_translates_with_the_torso_without_turning(views):
    env = _env()
    env._update_camera()
    displacement = torch.tensor([3.0, -4.0, 0.5])
    _move_torso(env, displacement)

    env._update_camera()
    eye, target = _view(views)
    # smoothed like the other follow cameras: a tenth of the way there on the first update
    assert torch.allclose(eye, torch.tensor(START_EYE) + 0.1 * displacement)
    assert torch.allclose(target, torch.tensor(START_LOOKAT) + 0.1 * displacement)

    for _ in range(200):
        env._update_camera()
    eye, target = _view(views)
    assert torch.allclose(eye, torch.tensor(START_EYE) + displacement, atol=1e-4)
    assert torch.allclose(target, torch.tensor(START_LOOKAT) + displacement, atol=1e-4)
    # the same offset from eye to target on every update: the view never turns
    for eye, target in views:
        assert torch.allclose(torch.tensor(eye) - torch.tensor(target), torch.tensor(START_EYE), atol=1e-4)


def test_camera_snaps_to_the_torso_after_a_reset_and_keeps_its_first_anchor(views):
    env = _env()
    env._update_camera()
    displacement = torch.tensor([-6.0, 2.0, 0.0])
    _move_torso(env, displacement)
    # what `_reset_idx` does when env 0 resets
    env._camera_eye_smooth = None
    env._camera_target_smooth = None

    env._update_camera()

    eye, target = _view(views)
    assert torch.allclose(eye, torch.tensor(START_EYE) + displacement)
    assert torch.allclose(target, torch.tensor(START_LOOKAT) + displacement)
