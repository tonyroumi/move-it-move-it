"""The one phase: each env's progress through the clip it tracks, observed only in the
BEHAVIOR_PHASE command slot. No simulator."""

import pytest
import torch

# imports MotionLearningEnv against IsaacLab stand-ins, so the package below imports without IsaacLab
from test_episode_reset_behaviors import NUM_ENVS, STEP_DT, _clip, _env, _spec

from moveitmoveit.mdp import BodyState
from moveitmoveit.mdp.observations import OBSERVATIONS
from moveitmoveit.motion.behavior import LOCOMOTION, behavior_commands, clip_phase
from moveitmoveit.motion.manifest import CommandIndex
from moveitmoveit.motion.motion_set import MotionSet

WALK, JAB = 0, 1
WALK_DURATION, JAB_DURATION = 2.0, 1.0
JAB_STEPS = round(JAB_DURATION / STEP_DT)


def _motions(tmp_path, loop: bool = False, random_start: bool = True) -> MotionSet:
    specs = [
        _spec("walk", _clip(tmp_path / "walk.npz", 61), 0.5, loop=loop, random_start=random_start),
        _spec("jab", _clip(tmp_path / "jab.npz", 31), 0.5, trigger="binary_key_0"),
    ]
    return MotionSet(specs, "cpu")


def _reset(motions: MotionSet, locomotion_phase: bool):
    """An env just reset, with play's keyboard as the only source of key presses."""
    env = _env(motions, resample_commands=False)
    env.cfg.locomotion_phase = locomotion_phase
    env._reset_idx(torch.arange(NUM_ENVS))
    env._update_behaviors()
    return env


def _step(env, steps: int = 1) -> None:
    for _ in range(steps):
        env.episode_length_buf += 1
        env._update_behaviors()


def _observed_phase(env) -> torch.Tensor:
    observed = behavior_commands(env.commands, env.motions.behavior_keys, env.behaviors.mode, env.behavior_phase)
    return observed[:, CommandIndex.BEHAVIOR_PHASE]


def _tracked_clip_times(env) -> torch.Tensor:
    """The clip time each env's tracking target is read at."""
    env.reference.tracking_targets = lambda motion_ids, times, yaw: times
    return env._current_tracking_targets()


# --- one definition -------------------------------------------------------------------------


def test_phase_is_clip_time_over_duration_bounded_and_finite_for_any_clock():
    times = torch.linspace(-1.0, 100.0, 1001)
    for duration in (0.0, 0.5, 3.0):
        phase = clip_phase(times, torch.full_like(times, duration))
        assert torch.isfinite(phase).all() and (phase >= 0.0).all() and (phase <= 1.0).all()
    assert torch.allclose(clip_phase(torch.tensor([0.0, 0.5, 1.5, 2.0]), torch.full((4,), 2.0)), torch.tensor([0.0, 0.25, 0.75, 1.0]))


def test_no_observation_term_carries_a_phase():
    # the phase is a command slot; nothing else in the observation holds one
    assert not [name for name in OBSERVATIONS if "phase" in name]
    assert not [name for name in BodyState._fields if "phase" in name]
    assert CommandIndex.BEHAVIOR_PHASE == 8


@pytest.mark.parametrize("locomotion_phase", [True, False])
@pytest.mark.parametrize("loop", [True, False])
def test_the_observed_phase_reads_the_clock_the_tracking_target_reads(tmp_path, locomotion_phase, loop):
    torch.manual_seed(0)
    motions = _motions(tmp_path, loop=loop)
    env = _reset(motions, locomotion_phase)
    # half the envs throw the jab, so both kinds of clip are tracked at once
    env.commands[::2, CommandIndex.BINARY_KEY_0] = 1.0

    for _ in range(3 * JAB_STEPS):
        _step(env)
        observes = env.behaviors.active | locomotion_phase
        expected = (_tracked_clip_times(env) / motions.library.duration[env.motion_ids]).clamp(0.0, 1.0)
        assert torch.allclose(_observed_phase(env)[observes], expected[observes])
        assert (_observed_phase(env)[~observes] == 0.0).all()


# --- in locomotion --------------------------------------------------------------------------


def test_locomotion_observes_no_phase_by_default(tmp_path):
    torch.manual_seed(0)
    env = _reset(_motions(tmp_path, loop=True), locomotion_phase=False)

    assert (env.motion_start_times > 0.0).any()
    for _ in range(2 * JAB_STEPS):
        assert (_observed_phase(env) == 0.0).all()
        _step(env)


def test_locomotion_phase_starts_at_the_reset_time_in_the_clip_and_advances_with_it(tmp_path):
    torch.manual_seed(0)
    env = _reset(_motions(tmp_path, loop=True), locomotion_phase=True)
    start_times = env.motion_start_times.clone()

    assert (start_times > 0.0).any()
    assert torch.allclose(_observed_phase(env), start_times / WALK_DURATION)

    _step(env, 12)
    expected = torch.remainder(start_times + 12 * STEP_DT, WALK_DURATION) / WALK_DURATION
    assert torch.allclose(_observed_phase(env), expected, atol=1e-5)


def test_a_looping_locomotion_clip_wraps_and_any_other_holds_at_1(tmp_path):
    steps = round(2.5 * WALK_DURATION / STEP_DT)

    (tmp_path / "loop").mkdir()
    looping = _reset(_motions(tmp_path / "loop", loop=True, random_start=False), locomotion_phase=True)
    _step(looping, steps)
    assert torch.allclose(_observed_phase(looping), torch.tensor(0.5), atol=1e-4)

    (tmp_path / "hold").mkdir()
    holding = _reset(_motions(tmp_path / "hold", loop=False, random_start=False), locomotion_phase=True)
    _step(holding, steps)
    assert (_observed_phase(holding) == 1.0).all()


# --- across a behavior ----------------------------------------------------------------------


@pytest.mark.parametrize("locomotion_phase", [True, False])
def test_a_running_behavior_observes_its_own_phase_whatever_the_locomotion_setting(tmp_path, locomotion_phase):
    env = _reset(_motions(tmp_path, random_start=False), locomotion_phase)
    _step(env, 6)

    env.commands[:, CommandIndex.BINARY_KEY_0] = 1.0
    _step(env)
    assert (env.motion_ids == JAB).all()
    assert (_observed_phase(env) == 0.0).all()

    _step(env, 9)
    assert torch.allclose(_observed_phase(env), torch.tensor(9 * STEP_DT / JAB_DURATION))


def test_the_phase_returns_to_the_locomotion_clip_when_the_behavior_completes(tmp_path):
    env = _reset(_motions(tmp_path, random_start=False), locomotion_phase=True)
    _step(env, 6)
    env.commands[:, CommandIndex.BINARY_KEY_0] = 1.0
    _step(env, JAB_STEPS)
    assert (env.motion_ids == JAB).all() and (_observed_phase(env) < 1.0).all()

    _step(env)

    # the walk's clock ran on underneath the jab
    assert (env.behaviors.mode == LOCOMOTION).all() and (env.motion_ids == WALK).all()
    assert torch.allclose(_observed_phase(env), torch.tensor((6 + JAB_STEPS + 1) * STEP_DT / WALK_DURATION))
