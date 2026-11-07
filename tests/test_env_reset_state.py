"""The reset root state the env computes, on a stand-in for the parts of the env it reads."""

from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("isaaclab.envs", reason="needs IsaacLab")

from moveitmoveit.env import MotionLearningEnv  # noqa: E402
from moveitmoveit.motion.reference import ResetFrames  # noqa: E402
from moveitmoveit.utils import transforms  # noqa: E402


@pytest.fixture
def frames() -> ResetFrames:
    return ResetFrames(
        dof_positions=torch.tensor([[0.1, 0.2]]),
        dof_velocities=torch.tensor([[0.3, 0.4]]),
        root_positions=torch.tensor([[1.0, 2.0, 0.9]]),
        root_rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
        root_linear_velocities=torch.tensor([[0.5, 0.0, 0.0]]),
        root_angular_velocities=torch.tensor([[0.0, 0.0, 0.25]]),
    )


@pytest.fixture
def env() -> SimpleNamespace:
    default_root_state = torch.zeros(3, 13)
    env_origins = torch.tensor([[0.0, 0.0, 0.0], [10.0, 20.0, 0.0], [30.0, 40.0, 0.0]])
    return SimpleNamespace(default_root_state=default_root_state, scene=SimpleNamespace(env_origins=env_origins))


def test_root_state_adds_env_origin_and_lift(env, frames):
    env_ids = torch.tensor([1])

    root_state, dof_pos, dof_vel = MotionLearningEnv._compute_robot_state(env, env_ids, frames)

    assert root_state[0, :3].tolist() == pytest.approx([11.0, 22.0, 1.0])
    assert root_state[0, 3:7].tolist() == [0.0, 0.0, 0.0, 1.0]
    assert root_state[0, 7:10].tolist() == [0.5, 0.0, 0.0]
    assert root_state[0, 10:13].tolist() == [0.0, 0.0, 0.25]
    assert torch.equal(dof_pos, frames.dof_positions)
    assert torch.equal(dof_vel, frames.dof_velocities)


def test_yaw_rotates_the_whole_root_position_including_env_origin(env, frames):
    env_ids = torch.tensor([1])
    yaw = transforms.random_yaw_orientation(1, "cpu")

    unrotated, _, _ = MotionLearningEnv._compute_robot_state(env, env_ids, frames)
    rotated, _, _ = MotionLearningEnv._compute_robot_state(env, env_ids, frames, yaw)

    assert torch.allclose(rotated[:, :3], transforms.quat_apply(yaw, unrotated[:, :3]))
    assert torch.allclose(rotated[:, 3:7], transforms.quat_mul(yaw, unrotated[:, 3:7]))
    assert torch.allclose(rotated[:, 7:10], transforms.quat_apply(yaw, unrotated[:, 7:10]))
    assert torch.allclose(rotated[:, 10:13], transforms.quat_apply(yaw, unrotated[:, 10:13]))
