"""The scripted viewer camera, on a stand-in for the parts of the env it reads.

play.py reaches _update_camera through the env step, play_motion.py calls it directly, so
both drive the same method exercised here.
"""

import math
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab.envs", reason="needs IsaacLab")

from moveitmoveit.env import MotionLearningEnv  # noqa: E402

ROOT = torch.tensor([3.0, -2.0, 0.9])
TARGET_OFFSET = torch.tensor([1.0, 0.0, 0.5])
EYE_OFFSET = torch.tensor([-6.0, 0.0, 3.0])


def yaw_quat(degrees: float) -> torch.Tensor:
    """xyzw rotation about +Z."""
    half = math.radians(degrees) / 2
    return torch.tensor([0.0, 0.0, math.sin(half), math.cos(half)])


def quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """xyzw Hamilton product a * b."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def rotate_z(vec: torch.Tensor, degrees: float) -> torch.Tensor:
    c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
    return torch.tensor([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1], vec[2]])


class Viewer:
    def __init__(self):
        self.views: list[tuple[tuple, tuple]] = []

    def set_camera_view(self, eye, target):
        self.views.append((eye, target))


@pytest.fixture(autouse=True)
def kit_viewport(monkeypatch) -> Viewer:
    """Stand in for the Kit viewport so no renderer is touched."""
    viewer = Viewer()
    module = ModuleType("isaaclab_physx.renderers.kit_viewport_utils")
    module.set_kit_renderer_camera_view = viewer.set_camera_view
    monkeypatch.setitem(sys.modules, module.__name__, module)
    return viewer


def make_env(camera_type: str, root_quat: torch.Tensor, root_pos: torch.Tensor = ROOT, num_envs: int = 1):
    root_pos_w = root_pos.unsqueeze(0).repeat(num_envs, 1)
    root_quat_w = root_quat.unsqueeze(0).repeat(num_envs, 1)
    random_yaw_quat = torch.zeros(num_envs, 4)
    random_yaw_quat[:, -1] = 1.0
    return SimpleNamespace(
        cfg=SimpleNamespace(camera_type=camera_type),
        robot=SimpleNamespace(
            data=SimpleNamespace(
                root_pos_w=SimpleNamespace(torch=root_pos_w),
                root_quat_w=SimpleNamespace(torch=root_quat_w),
            )
        ),
        random_yaw_quat=random_yaw_quat,
        _camera_target_offset=TARGET_OFFSET.clone(),
        _camera_eye_offset=EYE_OFFSET.clone(),
        _camera_smoothing_alpha=0.1,
        _camera_eye_smooth=None,
        _camera_target_smooth=None,
        sim=Viewer(),
    )


def set_robot(env, root_quat: torch.Tensor, root_pos: torch.Tensor = ROOT):
    env.robot.data.root_pos_w.torch[:] = root_pos
    env.robot.data.root_quat_w.torch[:] = root_quat


def view(env) -> tuple[torch.Tensor, torch.Tensor]:
    MotionLearningEnv._update_camera(env)
    eye, target = env.sim.views[-1]
    return torch.tensor(eye), torch.tensor(target)


def expected_view(yaw_degrees: float, root_pos: torch.Tensor = ROOT) -> tuple[torch.Tensor, torch.Tensor]:
    target = root_pos + rotate_z(TARGET_OFFSET, yaw_degrees)
    return target + rotate_z(EYE_OFFSET, yaw_degrees), target


@pytest.mark.parametrize("yaw", [0.0, 90.0, 180.0, 270.0, 360.0])
def test_third_person_stays_behind_the_robot_heading(yaw):
    env = make_env("third-person", yaw_quat(yaw))

    eye, target = view(env)

    expected_eye, expected_target = expected_view(yaw)
    assert torch.allclose(eye, expected_eye, atol=1e-5)
    assert torch.allclose(target, expected_target, atol=1e-5)
    # 5 m behind the heading along the ground, at the configured height
    heading = rotate_z(torch.tensor([1.0, 0.0, 0.0]), yaw)
    assert torch.allclose(eye - ROOT, -5.0 * heading + torch.tensor([0.0, 0.0, 3.5]), atol=1e-5)


def test_quaternions_are_read_xyzw():
    # +90 deg about +Z in xyzw: the robot faces +Y, so the camera sits on its -Y side
    env = make_env("third-person", torch.tensor([0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)]))

    eye, target = view(env)

    assert torch.allclose(target - ROOT, torch.tensor([0.0, 1.0, 0.5]), atol=1e-5)
    assert torch.allclose(eye - ROOT, torch.tensor([0.0, -5.0, 3.5]), atol=1e-5)


def test_negated_quaternion_gives_the_same_view():
    env = make_env("third-person", yaw_quat(135.0))
    flipped = make_env("third-person", -yaw_quat(135.0))

    assert all(torch.allclose(a, b, atol=1e-5) for a, b in zip(view(env), view(flipped)))


@pytest.mark.parametrize("tilt_axis", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
def test_body_tilt_does_not_pitch_or_roll_the_camera(tilt_axis):
    half = math.radians(25.0) / 2
    tilt = torch.tensor([*(math.sin(half) * a for a in tilt_axis), math.cos(half)])
    env = make_env("third-person", quat_mul(yaw_quat(60.0), tilt))

    eye, target = view(env)

    expected_eye, expected_target = expected_view(60.0)
    assert torch.allclose(eye, expected_eye, atol=1e-5)
    assert torch.allclose(target, expected_target, atol=1e-5)


def test_third_person_tracks_position():
    moved = ROOT + torch.tensor([4.0, 1.5, 0.2])
    env = make_env("third-person", yaw_quat(30.0), root_pos=moved)

    eye, target = view(env)

    expected_eye, expected_target = expected_view(30.0, root_pos=moved)
    assert torch.allclose(eye, expected_eye, atol=1e-5)
    assert torch.allclose(target, expected_target, atol=1e-5)


def test_full_turn_ends_behind_the_robot_through_smoothing():
    env = make_env("third-person", yaw_quat(0.0))
    view(env)

    for step in range(1, 361):
        set_robot(env, yaw_quat(float(step)))
        view(env)
    for _ in range(200):
        eye, target = view(env)

    expected_eye, expected_target = expected_view(360.0)
    assert torch.allclose(eye, expected_eye, atol=1e-3)
    assert torch.allclose(target, expected_target, atol=1e-3)


def test_half_turn_moves_the_camera_to_the_other_side():
    env = make_env("third-person", yaw_quat(0.0))
    start_eye, _ = view(env)

    set_robot(env, yaw_quat(180.0))
    for _ in range(200):
        eye, _ = view(env)

    # facing -X now, so the camera is on the +X side of the robot instead of the -X side
    assert start_eye[0] < ROOT[0] < eye[0]


def test_third_person_follows_env_zero_with_many_envs():
    env = make_env("third-person", yaw_quat(90.0), num_envs=3)
    env.robot.data.root_quat_w.torch[1:] = yaw_quat(0.0)

    eye, target = view(env)

    expected_eye, expected_target = expected_view(90.0)
    assert torch.allclose(eye, expected_eye, atol=1e-5)
    assert torch.allclose(target, expected_target, atol=1e-5)


def test_facing_camera_still_uses_the_reset_yaw(kit_viewport):
    env = make_env("facing", yaw_quat(90.0))
    env.random_yaw_quat[:] = yaw_quat(30.0)

    eye, target = view(env)

    expected_eye, expected_target = expected_view(30.0)
    assert torch.allclose(eye, expected_eye, atol=1e-5)
    assert torch.allclose(target, expected_target, atol=1e-5)
    assert kit_viewport.views[-1] == env.sim.views[-1]


def test_none_camera_does_nothing():
    env = make_env("none", yaw_quat(90.0))

    MotionLearningEnv._update_camera(env)

    assert env.sim.views == []
