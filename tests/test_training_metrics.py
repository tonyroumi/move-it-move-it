"""Training metrics: per-step reward means and zero/nonzero command tracking diagnostics.

Runs without IsaacLab: `isaaclab.envs` is always stubbed (the agent only names `DirectRLEnv` in
annotations), and `isaaclab.utils.math` is stubbed with xyzw quaternion math when IsaacLab is
absent. Stubs and the modules imported under them are dropped from sys.modules afterwards.
"""

import contextlib
import importlib
import math
import sys
import types
from unittest import mock

import pytest
import torch

LIN = "lin_vel"
YAW = "yaw_rate"


def _stub_math() -> types.ModuleType:
    def quat_conjugate(q):
        return torch.cat((-q[..., :3], q[..., 3:]), dim=-1)

    def quat_apply(q, v):
        xyz, w = q[..., :3], q[..., 3:]
        t = 2.0 * torch.cross(xyz, v, dim=-1)
        return v + w * t + torch.cross(xyz, t, dim=-1)

    def quat_apply_inverse(q, v):
        return quat_apply(quat_conjugate(q), v)

    def unused(*args, **kwargs):
        raise NotImplementedError("not stubbed")

    module = types.ModuleType("isaaclab.utils.math")
    module.quat_conjugate = quat_conjugate
    module.quat_apply = quat_apply
    module.quat_apply_inverse = quat_apply_inverse
    for name in ("euler_xyz_from_quat", "normalize", "quat_apply_yaw", "quat_mul", "random_yaw_orientation", "scale_transform"):
        setattr(module, name, unused)
    return module


@pytest.fixture(scope="module")
def m():
    with mock.patch.dict(sys.modules):
        has_isaaclab_math = _importable("isaaclab.utils.math")
        envs = types.ModuleType("isaaclab.envs")
        envs.DirectRLEnv = object
        sys.modules["isaaclab.envs"] = envs
        if not has_isaaclab_math:
            sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
            sys.modules["isaaclab.utils"] = types.ModuleType("isaaclab.utils")
            sys.modules["isaaclab.utils.math"] = _stub_math()

        # The stub math is plain Python, so let the TorchScript decorators in transforms pass it through.
        passthrough = mock.patch.object(torch.jit, "script", lambda fn: fn)
        with contextlib.nullcontext() if has_isaaclab_math else passthrough:
            from moveitmoveit import mdp
            from moveitmoveit.agents.base import BaseAgent
            from moveitmoveit.manifest import CommandIndex, load_manifest
            from moveitmoveit.utils import transforms
            from moveitmoveit.utils.logger import Logger

        yield types.SimpleNamespace(
            mdp=mdp,
            BaseAgent=BaseAgent,
            CommandIndex=CommandIndex,
            load_manifest=load_manifest,
            transforms=transforms,
            Logger=Logger,
        )


def _importable(name: str) -> bool:
    try:
        importlib.import_module(name)
    except Exception:
        return False
    return True


# --- env side: what one step reports ------------------------------------------------------------


def _metric(name: str, kind: str, bucket: str) -> str:
    unit = "m/s" if name == LIN else "rad/s"
    return f"{name}_{kind} ({bucket} cmd) [{unit}]"


def _nan_equal(actual: torch.Tensor, expected: list[float]) -> None:
    torch.testing.assert_close(actual, torch.tensor(expected), equal_nan=True)


def test_metrics_split_by_whether_the_command_is_zero(m):
    actual = torch.tensor([[0.3, -0.4], [0.5, 0.5], [2.0, 0.0], [1.0, 1.0]])
    commanded = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 0.5], [0.0, 0.0]])
    tracked = torch.tensor([True, True, True, False])
    out = m.mdp.velocity_tracking_metrics(LIN, actual, commanded, tracked, "m/s")
    nan = float("nan")

    _nan_equal(out[_metric(LIN, "error", "zero")], [0.5, nan, nan, nan])
    _nan_equal(out[_metric(LIN, "error", "nonzero")], [nan, math.hypot(0.5, 0.5), math.hypot(2.0, 0.5), nan])
    _nan_equal(out[_metric(LIN, "commanded", "nonzero")], [nan, 1.0, 0.5, nan])
    # projected onto the commanded direction: forward speed for a forward command, lateral for lateral
    _nan_equal(out[_metric(LIN, "actual_along_cmd", "nonzero")], [nan, 0.5, 0.0, nan])


def test_yaw_actual_along_command_is_signed_toward_the_commanded_turn(m):
    actual = torch.tensor([[0.4], [0.4], [-0.2]])
    commanded = torch.tensor([[-1.0], [1.0], [0.0]])
    out = m.mdp.velocity_tracking_metrics(YAW, actual, commanded, torch.ones(3, dtype=torch.bool), "rad/s")
    nan = float("nan")

    _nan_equal(out[_metric(YAW, "error", "nonzero")], [1.4, 0.6, nan])
    _nan_equal(out[_metric(YAW, "error", "zero")], [nan, nan, 0.2])
    _nan_equal(out[_metric(YAW, "commanded", "nonzero")], [1.0, 1.0, nan])
    _nan_equal(out[_metric(YAW, "actual_along_cmd", "nonzero")], [-0.4, 0.4, nan])


def _yaw_quat(yaw: float, pitch: float) -> torch.Tensor:
    """xyzw: yaw about Z after pitch about Y."""
    cy, sy, cp, sp = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(pitch / 2), math.sin(pitch / 2)
    return torch.tensor([-sy * sp, cy * sp, sy * cp, cy * cp])


def _context(m, commands: torch.Tensor, rotations: torch.Tensor):
    n = commands.shape[0]
    generator = torch.Generator().manual_seed(0)
    state = m.mdp.BodyState(
        torch.zeros(n, 28),
        torch.zeros(n, 28),
        torch.zeros(n, 3),
        rotations,
        torch.randn(n, 3, generator=generator),
        torch.randn(n, 3, generator=generator),
        torch.zeros(n, 4, 3),
    )
    return m.mdp.StepContext(state, torch.zeros(n, 28), commands, torch.zeros(n, 28), torch.zeros(n, 28), lambda: None)


@pytest.fixture
def specs(m, write_manifest):
    def _specs(motions: dict):
        return m.load_manifest(write_manifest({name: {"file": f"{name}.npz", **motion} for name, motion in motions.items()}))

    return _specs


def test_errors_use_the_tracking_rewards_frames_and_definitions(m, specs):
    composed = m.mdp.ComposedMDP(
        specs(
            {
                "walk": {"rewards": {"lin_vel_tracking": {"weight": 3.0, "scale": 2.0}, "yaw_vel_tracking": 0.0}},
                "box": {"rewards": {"joint_vel": 1.0}},
            }
        ),
        ("root_height",),
        ("root_height",),
        "cpu",
    )
    cmd = m.CommandIndex
    commands = torch.zeros(4, len(cmd))
    commands[0, [cmd.LIN_X, cmd.LIN_Y, cmd.YAW]] = torch.tensor([0.8, 0.2, -1.0])  # all nonzero
    commands[1, cmd.YAW] = 1.2  # turn in place: zero linear, nonzero yaw
    commands[2, cmd.LIN_X] = 0.5  # walk straight: nonzero linear, zero yaw
    commands[3, cmd.LIN_X] = 0.7  # a motion that doesn't track velocity
    rotations = torch.stack([_yaw_quat(0.3, 0.2), _yaw_quat(-1.0, 0.1), _yaw_quat(2.0, -0.3), _yaw_quat(0.0, 0.0)])
    c = _context(m, commands, rotations)
    motion_ids = torch.tensor([0, 0, 0, 1])

    metrics = composed.command_tracking(c, motion_ids)
    _, raw, _ = composed.rewards(c, motion_ids)

    for name, reward in ((LIN, "lin_vel_tracking"), (YAW, "yaw_vel_tracking")):
        error = torch.where(
            torch.isnan(metrics[_metric(name, "error", "zero")]),
            metrics[_metric(name, "error", "nonzero")],
            metrics[_metric(name, "error", "zero")],
        )
        # every velocity-tracking env lands in exactly one bucket; the other motion in neither
        assert torch.isnan(error).tolist() == [False, False, False, True]
        torch.testing.assert_close(torch.exp(-2.0 * error[:3] ** 2), raw[reward][:3])

    zero_lin = metrics[_metric(LIN, "error", "zero")]
    assert (~torch.isnan(zero_lin)).tolist() == [False, True, False, False]
    zero_yaw = metrics[_metric(YAW, "error", "zero")]
    assert (~torch.isnan(zero_yaw)).tolist() == [False, False, True, False]
    torch.testing.assert_close(zero_lin[1], torch.linalg.vector_norm(c.local_lin_vel_xy[1]))


def test_no_metrics_without_a_velocity_tracking_reward(m, specs):
    composed = m.mdp.ComposedMDP(specs({"goal": {"rewards": {"target_hit": 1.0}}}), ("root_height",), ("root_height",), "cpu")
    commands = torch.zeros(2, len(m.CommandIndex))
    rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(2, 1)
    assert composed.command_tracking(_context(m, commands, rotations), torch.zeros(2, dtype=torch.long)) == {}


# --- agent side: what gets aggregated and logged -------------------------------------------------


@pytest.fixture
def agent(m):
    class Agent(m.BaseAgent):
        _initialize_models = _initialize_optimizer = _initialize_storage = lambda *args: None
        act = process_env_step = update = inference = train = lambda *args, **kwargs: None
        write_checkpoint = load_checkpoint = lambda *args, **kwargs: None

    agent = Agent(cfg=None, logger=m.Logger(backend="none"))
    agent._step_dt = 0.5
    return agent


def _step(agent, rewards, dones, raw=None, tracking=None):
    rewards = torch.tensor(rewards)
    infos = {
        "reward_terms": {"task": 2.0 * rewards},
        "reward_terms_raw": {"lin_vel_tracking": torch.tensor(raw if raw is not None else rewards.tolist())},
    }
    if tracking is not None:
        infos["command_tracking"] = {name: torch.tensor(values) for name, values in tracking.items()}
    agent._track_episode_stats(rewards, torch.tensor(dones), infos)
    return agent.logger._tracking_data


def test_per_step_means_weigh_every_step_alike_across_episode_lengths(agent):
    # env 0: a 1-step episode earning 4.0; env 1: a 3-step episode earning 1.0 + 2.0 + 3.0
    logged = _step(agent, [4.0, 1.0], [True, False], raw=[0.5, 0.1])
    _step(agent, [9.0, 2.0], [False, False], raw=[0.9, 0.2])
    logged = _step(agent, [9.0, 3.0], [False, True], raw=[0.9, 0.3])

    # existing tags keep their per-episode-sum semantics
    assert logged["Performance/Episode Reward (mean)"] == pytest.approx((4.0 + 6.0) / 2)
    assert logged["Performance/Episode Length (mean)"] == pytest.approx(2.0)
    assert logged["Reward Terms/task"] == pytest.approx((8.0 + 12.0) / 2)
    assert logged["Reward Terms Raw/lin_vel_tracking"] == pytest.approx((0.5 + 0.6) / 2)
    assert agent.mean_episode_reward == pytest.approx(5.0)

    # per step: summed over the window's episodes, divided by their summed lengths
    assert logged["Performance/Episode Reward per Step (mean)"] == pytest.approx(10.0 / 4)
    assert logged["Reward Terms Per Step/task"] == pytest.approx(20.0 / 4)
    assert logged["Reward Terms Raw Per Step/lin_vel_tracking"] == pytest.approx(1.1 / 4)
    assert agent.logger._core_performance["Reward per Step (mean)"] == pytest.approx(2.5)


def test_tracking_buckets_pool_steps_across_mixed_command_episodes(agent):
    nan = float("nan")
    zero, nonzero = "lin_vel_error (zero cmd) [m/s]", "lin_vel_error (nonzero cmd) [m/s]"

    # env 0: one episode that is zero-commanded for 2 steps, then nonzero for 1, then ends;
    # env 1: nonzero-commanded throughout, still running at the end, so not counted.
    logged = _step(agent, [1.0, 1.0], [False, False], tracking={zero: [0.1, nan], nonzero: [nan, 5.0]})
    assert not any(tag.startswith("Command Tracking/") for tag in logged), "nothing is logged before an episode ends"
    _step(agent, [1.0, 1.0], [False, False], tracking={zero: [0.3, nan], nonzero: [nan, 5.0]})
    logged = _step(agent, [1.0, 1.0], [True, False], tracking={zero: [nan, nan], nonzero: [0.6, nan]})

    assert logged[f"Command Tracking/{zero}"] == pytest.approx(0.2)
    assert logged[f"Command Tracking/{nonzero}"] == pytest.approx(0.6)

    # env 1 ends after 3 nonzero steps of error 5.0 and one step neither bucket covers;
    # env 0 runs a 1-step nonzero episode
    logged = _step(agent, [1.0, 1.0], [True, True], tracking={zero: [nan, nan], nonzero: [1.4, 5.0]})
    # the zero bucket is unchanged: only env 0's first episode had zero-commanded steps
    assert logged[f"Command Tracking/{zero}"] == pytest.approx(0.2)
    # pooled by step, not by episode: (0.6 + 1.4 + 3 * 5.0) / 5
    assert logged[f"Command Tracking/{nonzero}"] == pytest.approx((0.6 + 1.4 + 15.0) / 5)


def test_an_empty_bucket_is_not_logged_as_zero(agent):
    nan = float("nan")
    zero, nonzero = "yaw_rate_error (zero cmd) [rad/s]", "yaw_rate_error (nonzero cmd) [rad/s]"

    logged = _step(agent, [1.0], [True], tracking={zero: [nan], nonzero: [0.7]})

    assert logged[f"Command Tracking/{nonzero}"] == pytest.approx(0.7)
    assert f"Command Tracking/{zero}" not in logged
