"""A manifest of behaviors alone, with no locomotion motion, runs them always on. No simulator."""

import math

import pytest
import torch

# imports MotionLearningEnv against IsaacLab stand-ins, so the package below imports without IsaacLab
from test_episode_reset_behaviors import NUM_ENVS, STEP_DT, _clip, _env, _spec

from moveitmoveit.motion.behavior import LOCOMOTION, behavior_commands
from moveitmoveit.motion.manifest import CommandIndex, load_manifest
from moveitmoveit.motion.motion_set import MotionSet

SPIN, KICK = 0, 1
SPIN_KEY, KICK_KEY = CommandIndex.BINARY_KEY_1, CommandIndex.BINARY_KEY_0
SPIN_DURATION, KICK_DURATION = 2.0, 1.0


def _behaviors(tmp_path, **kwargs) -> MotionSet:
    """Two behaviors and no locomotion motion. The spin is drawn three times as often as the kick,
    and its key slot is the higher one, so a motion id is not its behavior index."""
    specs = [
        _spec("spin", _clip(tmp_path / "spin.npz", 61), 0.75, trigger="binary_key_1", **kwargs),
        _spec("kick", _clip(tmp_path / "kick.npz", 31), 0.25, trigger="binary_key_0", **kwargs),
    ]
    return MotionSet(specs, "cpu")


def _manifest(tmp_path, motions: str) -> str:
    for name in ("walk", "spin", "kick"):
        _clip(tmp_path / f"{name}.npz", 31)
    path = tmp_path / "manifest.yaml"
    path.write_text("motions:\n" + motions.replace("FILE", str(tmp_path)))
    return str(path)


def _observed(env) -> torch.Tensor:
    return behavior_commands(env.commands, env.motions.behavior_keys, env.behaviors.mode, env.behavior_phase)


def _step(env, steps: int = 1) -> None:
    """The behavior update of each following step, as `_get_observations` runs it."""
    for _ in range(steps):
        env.episode_length_buf += 1
        env._update_behaviors()


# --- the manifest ---------------------------------------------------------------------------


def test_a_manifest_of_behaviors_alone_loads(tmp_path):
    path = _manifest(
        tmp_path,
        "  spin: {file: FILE/spin.npz, trigger: binary_key_1, sample_prob: 0.75}\n"
        "  kick: {file: FILE/kick.npz, trigger: binary_key_0, sample_prob: 0.25}\n",
    )

    specs = load_manifest(path)

    assert [spec.trigger for spec in specs] == ["binary_key_1", "binary_key_0"]
    assert MotionSet(specs, "cpu").always_on


def test_a_manifest_with_a_locomotion_motion_is_not_always_on(tmp_path):
    path = _manifest(
        tmp_path,
        "  walk: {file: FILE/walk.npz}\n  kick: {file: FILE/kick.npz, trigger: binary_key_0}\n",
    )
    assert not MotionSet(load_manifest(path), "cpu").always_on
    assert not MotionSet([_spec("walk", _clip(tmp_path / "only.npz", 31), 1.0)], "cpu").always_on


@pytest.mark.parametrize(
    "motions, error",
    [
        # a declared locomotion motion that no reset could start is still a mistake
        (
            "  walk: {file: FILE/walk.npz, sample_prob: 0}\n  kick: {file: FILE/kick.npz, trigger: binary_key_0}\n",
            "behaviors need a locomotion motion",
        ),
        (
            "  spin: {file: FILE/spin.npz, trigger: binary_key_0}\n  kick: {file: FILE/kick.npz, trigger: binary_key_0}\n",
            "each trigger more than one motion",
        ),
        ("  kick: {file: FILE/kick.npz, trigger: binary_key_0, sample_prob: 0}\n", "needs a positive sample_prob"),
        ("  kick: {file: FILE/kick.npz, trigger: binary_key_0, commands: {lin_x: 1.0}}\n", "declare none"),
        ("  kick: {file: FILE/kick.npz, trigger: binary_key_0, standing: {}}\n", "no standing mode"),
        ("  kick: {file: FILE/kick.npz, trigger: goal_x}\n", "kick.trigger"),
    ],
)
def test_the_other_trigger_checks_still_apply(tmp_path, motions, error):
    with pytest.raises(ValueError, match=error):
        load_manifest(_manifest(tmp_path, motions))


# --- the motion set's tables ----------------------------------------------------------------


def test_always_on_tables_are_finite(tmp_path):
    motions = _behaviors(tmp_path)

    assert motions.trigger_rate == 0.0
    assert torch.isfinite(motions.reset_sample_prob).all() and torch.isfinite(motions.trigger_weights).all()
    assert motions.reset_sample_prob.tolist() == pytest.approx([0.75, 0.25])
    # indexed by behavior (kick, spin): sample_prob / duration, normalized
    assert motions.trigger_weights.tolist() == pytest.approx([0.4, 0.6])


def test_tables_with_a_locomotion_motion_are_unchanged(tmp_path):
    specs = [
        _spec("walk", _clip(tmp_path / "walk.npz", 61), 0.5),
        _spec("stand", _clip(tmp_path / "stand.npz", 61), 0.25),
        _spec("kick", _clip(tmp_path / "kick.npz", 31), 0.25, trigger="binary_key_0"),
    ]
    motions = MotionSet(specs, "cpu")

    assert not motions.always_on
    assert motions.reset_sample_prob.tolist() == pytest.approx([2 / 3, 1 / 3, 0.0])
    assert motions.trigger_weights.tolist() == pytest.approx([1.0])
    # behavior share / (locomotion share * mean behavior duration)
    assert motions.trigger_rate == pytest.approx(0.25 / (0.75 * KICK_DURATION))


def test_reset_draws_behaviors_by_sample_prob(tmp_path):
    torch.manual_seed(0)
    motions = _behaviors(tmp_path)

    clips = motions.sample_reset(40_000)

    share = torch.bincount(clips.motion_ids, minlength=2) / clips.motion_ids.shape[0]
    assert share.tolist() == pytest.approx([0.75, 0.25], abs=0.01)
    assert (clips.times >= 0.0).all() and (clips.times <= motions.library.duration[clips.motion_ids]).all()
    assert (clips.times > 0.0).any()


def test_restarts_draw_behaviors_by_trigger_weights(tmp_path):
    torch.manual_seed(0)
    motions = _behaviors(tmp_path)

    behaviors = motions.sample_behaviors(40_000)

    share = torch.bincount(behaviors, minlength=2) / behaviors.shape[0]
    assert share.tolist() == pytest.approx(motions.trigger_weights.tolist(), abs=0.01)


# --- the env --------------------------------------------------------------------------------


@pytest.mark.parametrize("resample_commands", [True, False])
def test_reset_starts_in_a_behavior_with_its_command_and_phase_on(tmp_path, resample_commands):
    torch.manual_seed(0)
    motions = _behaviors(tmp_path)
    env = _env(motions, resample_commands)

    env._reset_idx(torch.arange(NUM_ENVS))
    start_times = env.motion_start_times.clone()

    assert (env.behaviors.mode != LOCOMOTION).all()
    assert (motions.behavior_motion_ids[env.behaviors.mode] == env.motion_ids).all()
    assert (env.motion_ids == SPIN).any() and (env.motion_ids == KICK).any()
    # the env is as far through the behavior as the reset placed it in the clip
    phase = start_times / motions.library.duration[env.motion_ids]
    assert torch.allclose(env.behavior_phase, phase) and (phase > 0.0).any()

    env._update_behaviors()
    observed = _observed(env)

    assert torch.allclose(observed[:, CommandIndex.BEHAVIOR_PHASE], phase)
    spin = env.motion_ids == SPIN
    assert (observed[spin, SPIN_KEY] == 1.0).all() and (observed[spin, KICK_KEY] == 0.0).all()
    assert (observed[~spin, KICK_KEY] == 1.0).all() and (observed[~spin, SPIN_KEY] == 0.0).all()
    # the raw key slots stay released: nothing pressed a key to get here
    assert (env.commands[:, motions.behavior_keys] == 0.0).all()


def test_phase_advances_with_the_clip_from_time_zero_without_random_start(tmp_path):
    motions = _behaviors(tmp_path, random_start=False)
    env = _env(motions, resample_commands=True)

    env._reset_idx(torch.arange(NUM_ENVS))
    assert (env.behavior_phase == 0.0).all()

    _step(env, 15)

    expected = 15 * STEP_DT / motions.library.duration[env.motion_ids]
    assert torch.allclose(env.behavior_phase, expected)
    assert torch.allclose(_observed(env)[:, CommandIndex.BEHAVIOR_PHASE], expected)


def test_a_completed_behavior_starts_another_on_the_same_step(tmp_path):
    torch.manual_seed(0)
    motions = _behaviors(tmp_path, random_start=False)
    env = _env(motions, resample_commands=True)
    env._reset_idx(torch.arange(NUM_ENVS))
    first = env.motion_ids.clone()
    kick_steps = round(KICK_DURATION / STEP_DT)

    # one step short of the kick's end nothing has completed
    _step(env, kick_steps - 1)
    assert (env.motion_ids == first).all() and (env.behavior_phase < 1.0).all()

    _step(env)
    kicked = first == KICK
    # the kicks completed and each started a newly drawn behavior at phase 0; the spins run on
    assert (env.behavior_phase[kicked] == 0.0).all()
    assert (env.behaviors.start_time[kicked] == kick_steps * STEP_DT).all()
    assert (env.motion_ids[kicked] == SPIN).any() and (env.motion_ids[kicked] == KICK).any()
    assert (env.motion_ids[~kicked] == SPIN).all()
    assert torch.allclose(env.behavior_phase[~kicked], torch.tensor(KICK_DURATION / SPIN_DURATION))


def test_every_env_runs_a_behavior_at_every_step(tmp_path):
    torch.manual_seed(0)
    motions = _behaviors(tmp_path)
    env = _env(motions, resample_commands=True)
    env._reset_idx(torch.arange(NUM_ENVS))

    for _ in range(round(4 * SPIN_DURATION / STEP_DT)):
        _step(env)
        observed = _observed(env)
        assert torch.isfinite(observed).all()
        assert (env.behaviors.mode != LOCOMOTION).all()
        assert (motions.behavior_motion_ids[env.behaviors.mode] == env.motion_ids).all()
        assert (observed[:, motions.behavior_keys].sum(dim=-1) == 1.0).all()
        phase = observed[:, CommandIndex.BEHAVIOR_PHASE]
        assert (phase >= 0.0).all() and (phase < 1.0).all()


def test_a_settled_behavior_starts_another(tmp_path):
    torch.manual_seed(0)
    specs = [
        _spec("kick", _clip(tmp_path / "kick.npz", 31), 1.0, trigger="binary_key_0", random_start=False),
    ]
    env = _env(MotionSet(specs, "cpu"), resample_commands=True)
    # the stand-in robot's joints move at 1 rad/s, under this bound, so the kick settles at half way
    env.motions.settle_min_phase[:] = 0.5
    env.motions.settle_max_dof_speed[:] = 2.0
    env._reset_idx(torch.arange(NUM_ENVS))

    _step(env, math.ceil(0.5 * KICK_DURATION / STEP_DT))

    assert (env.behaviors.mode == 0).all()
    assert (env.behavior_phase == 0.0).all()


def test_keys_neither_start_nor_switch_a_behavior(tmp_path):
    # play's keyboard writes the key slots itself
    motions = _behaviors(tmp_path, random_start=False)
    env = _env(motions, resample_commands=False)
    env._reset_idx(torch.arange(NUM_ENVS))
    running = env.motion_ids.clone()

    env.commands[:, motions.behavior_keys] = 1.0
    _step(env, 5)

    assert (env.motion_ids == running).all()
    assert torch.allclose(env.behavior_phase, 5 * STEP_DT / motions.library.duration[running])


def test_a_partial_reset_leaves_the_other_envs_running(tmp_path):
    torch.manual_seed(0)
    motions = _behaviors(tmp_path, random_start=False)
    env = _env(motions, resample_commands=True)
    env._reset_idx(torch.arange(NUM_ENVS))
    _step(env, 10)
    running, phase = env.motion_ids.clone(), env.behavior_phase.clone()

    reset = torch.arange(0, NUM_ENVS, 2)
    env._reset_idx(reset)

    assert (env.behavior_phase[reset] == 0.0).all()
    assert (env.behaviors.mode[reset] != LOCOMOTION).all()
    kept = torch.arange(1, NUM_ENVS, 2)
    assert (env.motion_ids[kept] == running[kept]).all()
    assert torch.equal(env.behavior_phase[kept], phase[kept])
