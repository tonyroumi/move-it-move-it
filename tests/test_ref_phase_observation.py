"""The opt-in `ref_phase` observation: progress through the reference clip each env tracks. No simulator."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

# imports MotionLearningEnv against IsaacLab stand-ins, so the package below imports without IsaacLab
from test_episode_reset_behaviors import MotionLearningEnv

from moveitmoveit.mdp import BodyState, ComposedMDP, observation_dim
from moveitmoveit.mdp.observations import OBSERVATIONS, reference_phase
from moveitmoveit.motion.manifest import COMMAND_DIM

DEFAULT_OBS = ("dof_pos", "dof_vel", "root_height", "root_orientation", "root_lin_vel", "root_ang_vel", "key_body_offsets")
# the default terms whose world-frame values call no quaternion op, which the stand-ins can't run
COMPOSED_OBS = tuple(name for name in DEFAULT_OBS if name != "root_orientation")
STEP_DT = 1.0 / 30.0


def _state(n: int, phase: torch.Tensor | None = None) -> BodyState:
    rand = torch.rand
    return BodyState(rand(n, 28), rand(n, 28), rand(n, 3), rand(n, 4), rand(n, 3), rand(n, 3), rand(n, 4, 3), phase)


def _env_phase(start_times, steps: int, duration: float, loop: bool) -> torch.Tensor:
    """`MotionLearningEnv._reference_phase` on just the state it reads."""
    n = start_times.shape[0]
    env = SimpleNamespace(
        episode_length_buf=torch.full((n,), steps, dtype=torch.long),
        step_dt=STEP_DT,
        motion_start_times=start_times,
        base_motion_ids=torch.zeros(n, dtype=torch.long),
        motions=SimpleNamespace(
            library=SimpleNamespace(duration=torch.tensor([duration])), loop=torch.tensor([loop])
        ),
    )
    return MotionLearningEnv._reference_phase(env)


def _default_selection(name: str) -> tuple[str, ...]:
    """A default observation selection, read from env_cfg.py, which builds IsaacLab config classes."""
    tree = ast.parse((Path(__file__).parents[1] / "src/moveitmoveit/env_cfg.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise AssertionError(f"env_cfg.py has no {name}")


def test_ref_phase_is_registered_with_dim_1_and_off_by_default():
    assert OBSERVATIONS["ref_phase"].dim == 1
    assert _default_selection("policy_obs") == DEFAULT_OBS
    assert _default_selection("style_obs") == DEFAULT_OBS


@pytest.mark.parametrize("loop", [True, False])
def test_phase_tracks_clip_time_over_duration_monotonically_within_a_clip(loop):
    duration = 2.0
    steps = torch.arange(0, int(duration / STEP_DT))  # one pass through the clip, short of its end
    phase = torch.stack([_env_phase(torch.zeros(1), int(s), duration, loop)[0] for s in steps])

    assert torch.allclose(phase, steps.float() * STEP_DT / duration)
    assert (phase[1:] > phase[:-1]).all()
    assert phase[0] == 0.0 and (phase < 1.0).all()


def test_phase_starts_at_the_reset_time_in_the_clip():
    start_times = torch.tensor([0.0, 0.5, 1.5])
    assert torch.allclose(_env_phase(start_times, 0, 2.0, loop=False), start_times / 2.0)


def test_looping_clip_wraps_and_other_clips_hold_at_1():
    start = torch.tensor([1.5])
    steps = 30  # 1 s on: 2.5 s into a 2 s clip
    assert torch.allclose(_env_phase(start, steps, 2.0, loop=True), torch.tensor([0.25]))
    assert torch.allclose(_env_phase(start, steps, 2.0, loop=False), torch.tensor([1.0]))


def test_phase_is_bounded_and_finite_for_any_clock():
    times = torch.linspace(-1.0, 100.0, 1001)
    for loop in (True, False):
        for duration in (0.0, 0.5, 3.0):
            phase = reference_phase(times, torch.zeros_like(times), torch.full_like(times, duration), torch.full_like(times, loop, dtype=torch.bool))
            assert torch.isfinite(phase).all() and (phase >= 0.0).all() and (phase <= 1.0).all()


def test_reference_phase_reads_nothing_of_the_behavior_system():
    # the env's phase source touches only the reset clip's clock, never behaviors or behavior_phase
    names = MotionLearningEnv._reference_phase.__code__.co_names
    assert not {"behaviors", "behavior_phase", "motion_ids", "start_time"} & set(names)


def test_selecting_ref_phase_grows_the_policy_observation_by_its_dim():
    with_phase = (*DEFAULT_OBS, "ref_phase")
    assert observation_dim(with_phase) - observation_dim(DEFAULT_OBS) == OBSERVATIONS["ref_phase"].dim == 1

    n = 5
    phase = torch.rand(n)
    state = _state(n, phase)
    commands = torch.zeros(n, COMMAND_DIM)

    composed = (*COMPOSED_OBS, "ref_phase")
    policy, style = ComposedMDP([], composed, COMPOSED_OBS, "cpu").observations(state, commands, local_frame=False)
    base_policy, base_style = ComposedMDP([], COMPOSED_OBS, COMPOSED_OBS, "cpu").observations(state, commands, local_frame=False)

    # env.py's observation_space and amp_observation_space
    assert policy.shape[-1] == observation_dim(composed) + COMMAND_DIM == base_policy.shape[-1] + 1
    assert style.shape[-1] == base_style.shape[-1] == observation_dim(COMPOSED_OBS)
    # the phase sits where it is listed, ahead of the trailing commands block
    assert torch.equal(policy[:, observation_dim(COMPOSED_OBS)], phase)
    assert torch.equal(policy[:, -COMMAND_DIM:], commands)


def test_default_selection_ignores_the_phase():
    n = 3
    commands = torch.zeros(n, COMMAND_DIM)
    state = _state(n)
    mdp = ComposedMDP([], COMPOSED_OBS, COMPOSED_OBS, "cpu")
    without, _ = mdp.observations(state, commands, local_frame=False)
    with_, _ = mdp.observations(state._replace(ref_phase=torch.rand(n)), commands, local_frame=False)
    assert torch.equal(without, with_)


def test_selecting_ref_phase_without_a_phase_fails_loudly():
    mdp = ComposedMDP([], ("ref_phase",), (), "cpu")
    with pytest.raises(ValueError, match="ref_phase"):
        mdp.observations(_state(2), torch.zeros(2, COMMAND_DIM), local_frame=False)


def test_ref_phase_is_rejected_as_a_style_observation():
    with pytest.raises(ValueError, match="style_obs"):
        ComposedMDP([], DEFAULT_OBS, (*DEFAULT_OBS, "ref_phase"), "cpu")
