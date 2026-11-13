"""An episode reset leaves every reset env in locomotion with no behavior or key active. No simulator."""

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def _import_env_without_isaaclab() -> type:
    """MotionLearningEnv, imported against stand-ins for the IsaacLab modules it needs.

    IsaacLab is installed only beside Isaac Sim, and importing it can start the simulator, so the
    stand-ins replace it here and leave sys.modules as they found it once the env is imported. The
    reset path calls none of the quaternion ops: the tests leave random_reset off and replace the
    robot state writes.
    """

    # typed like IsaacLab's, because `utils.transforms` scripts functions that call them
    def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        raise AssertionError("an IsaacLab stand-in was called")

    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        raise AssertionError("an IsaacLab stand-in was called")

    def unused(*args, **kwargs):
        raise AssertionError("an IsaacLab stand-in was called")

    class DirectRLEnv:
        def _reset_idx(self, env_ids: torch.Tensor) -> None:
            self.episode_length_buf[env_ids] = 0

    math = sys.modules.get("isaaclab.utils.math") or types.ModuleType("isaaclab.utils.math")
    functions = {"quat_apply": quat_apply, "quat_mul": quat_mul}
    # every name the package imports from isaaclab.utils.math
    names = (
        "euler_xyz_from_quat",
        "normalize",
        "quat_apply",
        "quat_apply_inverse",
        "quat_apply_yaw",
        "quat_conjugate",
        "quat_mul",
        "random_yaw_orientation",
        "scale_transform",
        "yaw_quat",
    )
    for name in names:
        if not hasattr(math, name):
            setattr(math, name, functions.get(name, unused))
    envs = types.ModuleType("isaaclab.envs")
    envs.DirectRLEnv = DirectRLEnv
    stand_ins = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.envs": envs,
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math,
        # env.py imports its config, which builds IsaacLab config classes; the reset path never reads it
        "moveitmoveit.env_cfg": types.SimpleNamespace(MotionLearningEnvCfg=object),
    }
    added = [name for name in stand_ins if name not in sys.modules]
    sys.modules |= {name: stand_ins[name] for name in added}
    try:
        return importlib.import_module("moveitmoveit.env").MotionLearningEnv
    finally:
        for name in added:
            del sys.modules[name]


MotionLearningEnv = _import_env_without_isaaclab()

from moveitmoveit.motion.behavior import LOCOMOTION, BehaviorManager, RandomKeyPresses, behavior_commands  # noqa: E402
from moveitmoveit.motion.manifest import CommandIndex, MotionSpec  # noqa: E402
from moveitmoveit.motion.motion_set import MotionSet  # noqa: E402

NUM_ENVS = 64
STEP_DT = 1.0 / 30.0
WALK, JAB = 0, 1


def _clip(path, num_frames: int) -> str:
    np.savez(
        path,
        dof_names=np.array(["hip"]),
        body_names=np.array(["torso"]),
        fps=30.0,
        dof_positions=np.zeros((num_frames, 1)),
        dof_velocities=np.zeros((num_frames, 1)),
        body_positions=np.zeros((num_frames, 1, 3)),
        body_rotations=np.tile([1.0, 0.0, 0.0, 0.0], (num_frames, 1, 1)),
        body_linear_velocities=np.zeros((num_frames, 1, 3)),
        body_angular_velocities=np.zeros((num_frames, 1, 3)),
    )
    return str(path)


def _spec(name: str, file: str, sample_prob: float, **kwargs) -> MotionSpec:
    return MotionSpec(name, file, sample_prob, {}, 0.0, {}, {}, **kwargs)


@pytest.fixture
def motions(tmp_path) -> MotionSet:
    # the jab carries almost all the selection weight, so a reset that could pick it nearly always would
    specs = [
        _spec("walk", _clip(tmp_path / "walk.npz", 61), 1e-3),
        _spec("jab", _clip(tmp_path / "jab.npz", 31), 1.0, trigger="binary_key_0"),
    ]
    return MotionSet(specs, "cpu")


def _env(motions: MotionSet, resample_commands: bool) -> MotionLearningEnv:
    """A MotionLearningEnv with the state its reset and behavior update read, and no simulator."""
    env = MotionLearningEnv.__new__(MotionLearningEnv)
    env.cfg = SimpleNamespace(random_reset=False, resample_commands=resample_commands, camera_type="none")
    env.render_enabled = False
    env.num_envs = NUM_ENVS
    env.device = "cpu"
    env.step_dt = STEP_DT
    env.episode_length_buf = torch.full((NUM_ENVS,), 90, dtype=torch.long)
    env.motions = motions
    env.robot = SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=torch.ones(NUM_ENVS, 1))))
    env.reference = SimpleNamespace(
        frames_for_reset=lambda motion_ids, times: None,
        amp_seed=lambda motion_ids, times, yaw: torch.zeros(motion_ids.shape[0], 1, 1),
    )
    env._compute_robot_state = lambda env_ids, frames, yaw=None: (None, None, None)
    env._write_robot_state = lambda *args: None

    env.actions = torch.zeros(NUM_ENVS, 1)
    env.previous_actions = torch.zeros(NUM_ENVS, 1)
    env.random_yaw_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(NUM_ENVS, 1)
    env.amp_observation_buffer = torch.zeros(NUM_ENVS, 1, 1)
    env.commands = torch.zeros(NUM_ENVS, len(CommandIndex))
    env.motion_ids = torch.full((NUM_ENVS,), WALK, dtype=torch.long)
    env.base_motion_ids = torch.full((NUM_ENVS,), WALK, dtype=torch.long)
    env.motion_start_times = torch.zeros(NUM_ENVS)
    env.behaviors = BehaviorManager(NUM_ENVS, motions.num_behaviors, "cpu")
    env.behavior_phase = torch.zeros(NUM_ENVS)
    # a press probability too small to fire, so only the presses a test sets up happen
    env.key_presses = RandomKeyPresses(NUM_ENVS, 1e-12, motions.trigger_weights, 10, STEP_DT, "cpu")
    return env


def _hold_jab(env: MotionLearningEnv) -> None:
    """Every env is mid-jab, with its key held: by training's key presses and in the command slot."""
    env.key_presses._behavior[:] = 0
    env.key_presses._hold[:] = 5
    env.key_presses._was_held[:] = True
    env.commands[:, CommandIndex.BINARY_KEY_0] = 1.0
    env.behaviors.mode[:] = 0
    env.behaviors.start_time[:] = 2.9
    env.motion_ids[:] = JAB


def _observed_behavior(env: MotionLearningEnv) -> torch.Tensor:
    observed = behavior_commands(env.commands, env.motions.behavior_keys, env.behaviors.mode, env.behavior_phase)
    return observed[:, env.motions.behavior_keys]


def _reset_and_step(env: MotionLearningEnv, steps: int = 1) -> None:
    """Reset every env, then run the behavior update of each following step without new presses."""
    env._reset_idx(torch.arange(NUM_ENVS))
    for _ in range(steps):
        env._update_behaviors()
        env.episode_length_buf += 1


@pytest.mark.parametrize("resample_commands", [True, False])
def test_reset_starts_in_locomotion_not_part_way_through_a_behavior(motions, resample_commands):
    env = _env(motions, resample_commands)
    _hold_jab(env)

    _reset_and_step(env)

    assert (env.behaviors.mode == LOCOMOTION).all()
    assert (env.motion_ids == WALK).all() and (env.base_motion_ids == WALK).all()
    assert (env.behavior_phase == 0.0).all()
    assert (_observed_behavior(env) == 0.0).all()


@pytest.mark.parametrize("resample_commands", [True, False])
def test_phase_stays_zero_after_reset(motions, resample_commands):
    env = _env(motions, resample_commands)
    _hold_jab(env)

    env._reset_idx(torch.arange(NUM_ENVS))
    assert (env.behavior_phase == 0.0).all()
    for _ in range(20):
        env._update_behaviors()
        env.episode_length_buf += 1
        assert (env.behaviors.mode == LOCOMOTION).all()
        assert (env.behavior_phase == 0.0).all()


@pytest.mark.parametrize("resample_commands", [True, False])
def test_held_key_is_released_on_reset(motions, resample_commands):
    env = _env(motions, resample_commands)
    _hold_jab(env)

    env._reset_idx(torch.arange(NUM_ENVS))

    assert (env.commands[:, motions.behavior_keys] == 0.0).all()
    assert (env.key_presses._hold == 0).all() and not env.key_presses._was_held.any()
    # training's key presses release the jab, so the next step holds no key
    if resample_commands:
        assert (env.key_presses.step() == 0.0).all()


def test_key_held_through_reset_does_not_start_a_behavior(motions):
    # play's keyboard writes the commands itself and keeps a key held across the reset
    env = _env(motions, resample_commands=False)
    _hold_jab(env)

    env._reset_idx(torch.arange(NUM_ENVS))
    env.commands[:, CommandIndex.BINARY_KEY_0] = 1.0
    env._update_behaviors()

    assert (env.behaviors.mode == LOCOMOTION).all()
    assert (env.behavior_phase == 0.0).all()


def test_press_after_reset_starts_a_behavior(motions):
    env = _env(motions, resample_commands=True)
    _hold_jab(env)

    env._reset_idx(torch.arange(NUM_ENVS))
    # training's key presses press the jab on the first step after the reset, a new press
    env.key_presses._press_prob = 1.0
    env._update_behaviors()

    assert (env.behaviors.mode == 0).all()
    assert (env.motion_ids == JAB).all()


def test_reset_samples_only_locomotion_clips_with_random_start(tmp_path):
    specs = [
        _spec("walk", _clip(tmp_path / "walk.npz", 61), 1e-3),
        _spec("stand", _clip(tmp_path / "stand.npz", 61), 1e-3, random_start=False),
        _spec("jab", _clip(tmp_path / "jab.npz", 31), 1.0, trigger="binary_key_0"),
    ]
    motions = MotionSet(specs, "cpu")

    clips = motions.sample_reset(4096)

    assert (motions.motion_behavior[clips.motion_ids] == LOCOMOTION).all()
    walk, stand = clips.motion_ids == 0, clips.motion_ids == 1
    assert walk.any() and stand.any()
    assert (clips.times[stand] == 0.0).all()
    assert (clips.times[walk] > 0.0).any() and (clips.times[walk] <= motions.library.duration[0]).all()


def test_reset_without_behaviors(tmp_path):
    motions = MotionSet([_spec("walk", _clip(tmp_path / "walk.npz", 61), 1.0)], "cpu")
    env = _env(motions, resample_commands=False)
    env.commands[:] = 1.0

    _reset_and_step(env)

    assert (env.behaviors.mode == LOCOMOTION).all()
    assert (env.motion_ids == WALK).all()
    assert (env.behavior_phase == 0.0).all()
    # with no behavior there is no key slot, so the reset leaves every command as it was
    assert (env.commands == 1.0).all()
