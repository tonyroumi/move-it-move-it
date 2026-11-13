"""The env's command and behavior sampling switches, which HIL sets from its config. No simulator."""

from types import SimpleNamespace

import pytest
import torch

from test_episode_reset_behaviors import JAB, NUM_ENVS, WALK, MotionLearningEnv, _clip, _env, _spec

from moveitmoveit.motion.behavior import LOCOMOTION
from moveitmoveit.motion.manifest import CommandIndex
from moveitmoveit.motion.motion_set import MotionSet


@pytest.fixture
def motions(tmp_path) -> MotionSet:
    specs = [
        _spec("walk", _clip(tmp_path / "walk.npz", 61), 1.0),
        _spec("jab", _clip(tmp_path / "jab.npz", 31), 1.0, trigger="binary_key_0"),
    ]
    return MotionSet(specs, "cpu")


def test_sampling_is_on_by_default():
    assert MotionLearningEnv.sample_commands and MotionLearningEnv.sample_behaviors


@pytest.mark.parametrize("sample_behaviors", [True, False])
def test_behavior_sampling_off_presses_no_key(motions, sample_behaviors):
    env = _env(motions, resample_commands=True)
    env.sample_behaviors = sample_behaviors
    env._reset_idx(torch.arange(NUM_ENVS))
    env.key_presses._press_prob = 1.0

    env._update_behaviors()

    if sample_behaviors:
        assert (env.behaviors.mode == 0).all() and (env.motion_ids == JAB).all()
    else:
        assert (env.behaviors.mode == LOCOMOTION).all() and (env.motion_ids == WALK).all()
        assert (env.commands[:, motions.behavior_keys] == 0).all()


def _command_env(sample_commands: bool, episode_length: int) -> MotionLearningEnv:
    env = MotionLearningEnv.__new__(MotionLearningEnv)
    env.cfg = SimpleNamespace(resample_commands=True, command_resample_steps=150)
    env.sample_commands = sample_commands
    env.episode_length_buf = torch.full((NUM_ENVS,), episode_length, dtype=torch.long)
    env.commands = torch.full((NUM_ENVS, len(CommandIndex)), 0.25)
    env.base_motion_ids = torch.zeros(NUM_ENVS, dtype=torch.long)
    env.ref_body_index = 0
    env.robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=SimpleNamespace(torch=torch.zeros(NUM_ENVS, 1, 3))))
    env.motions = SimpleNamespace(sample_commands=lambda ids, xy: torch.ones(ids.shape[0], len(CommandIndex)))
    env.forgotten = []
    env.mdp = SimpleNamespace(reset_memory=env.forgotten.append)
    return env


@pytest.mark.parametrize("episode_length", [0, 151])
def test_command_sampling_off_draws_no_command(episode_length):
    on, off = _command_env(True, episode_length), _command_env(False, episode_length)

    on._resample_commands()
    off._resample_commands()

    assert (on.commands == 1.0).all()
    assert (off.commands == 0.25).all()
    # a new episode still forgets what terms remembered; a periodic resample point without a draw does not
    assert len(on.forgotten) == 1
    assert len(off.forgotten) == (1 if episode_length == 0 else 0)
