"""Composition semantics: selection, per-motion tables, lazy reference sampling, default layout."""

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from conftest import REPO_ROOT  # noqa: E402
from moveitmoveit.manifest import CommandIndex, load_manifest  # noqa: E402
from moveitmoveit.mdp import (  # noqa: E402
    OBSERVATIONS,
    REWARDS,
    TERMINATIONS,
    BodyState,
    ComposedMDP,
    StepContext,
    observation_dim,
)
from moveitmoveit.utils import transforms  # noqa: E402

NUM_ENVS = 6
DEFAULT_OBS = ("dof_pos", "dof_vel", "root_height", "root_orientation", "root_lin_vel", "root_ang_vel", "key_body_offsets")


@pytest.fixture
def specs(write_manifest):
    def _specs(motions: dict):
        return load_manifest(write_manifest({name: {"file": f"{name}.npz", **motion} for name, motion in motions.items()}))

    return _specs


def state(num_dofs: int = 28) -> BodyState:
    return BodyState(
        torch.randn(NUM_ENVS, num_dofs),
        torch.randn(NUM_ENVS, num_dofs),
        torch.randn(NUM_ENVS, 3),
        torch.nn.functional.normalize(torch.randn(NUM_ENVS, 4), dim=-1),
        torch.randn(NUM_ENVS, 3),
        torch.randn(NUM_ENVS, 3),
        torch.randn(NUM_ENVS, 4, 3),
    )


def context(targets=None) -> StepContext:
    def sample_targets():
        if targets is None:
            raise AssertionError("the reference motion was sampled, but no selected term needs it")
        return targets

    return StepContext(
        state(),
        torch.randn(NUM_ENVS, 28),
        torch.randn(NUM_ENVS, len(CommandIndex)),
        torch.randn(NUM_ENVS, 28),
        torch.randn(NUM_ENVS, 28),
        sample_targets,
    )


def compute_proprioceptive_obs(s: BodyState, local_frame: bool) -> torch.Tensor:
    """The observation exactly as observations.py built it, kept here as the layout's golden copy."""
    num_envs, num_bodies = s.root_positions.shape[0], s.key_body_positions.shape[1]
    body_offsets = s.key_body_positions - s.root_positions.unsqueeze(1)
    if local_frame:
        lin_vel = transforms.quat_apply_inverse(s.root_rotations, s.root_linear_velocities)
        ang_vel = transforms.quat_apply_inverse(s.root_rotations, s.root_angular_velocities)
        rotations = s.root_rotations.unsqueeze(1).expand(num_envs, num_bodies, 4)
        offsets = transforms.quat_apply_inverse(rotations, body_offsets).reshape(num_envs, -1)
    else:
        lin_vel, ang_vel = s.root_linear_velocities, s.root_angular_velocities
        offsets = body_offsets.reshape(num_envs, -1)
    return torch.cat(
        (
            s.dof_positions,
            s.dof_velocities,
            s.root_positions[:, 2:3],
            transforms.quaternion_to_tangent_and_normal(s.root_rotations),
            lin_vel,
            ang_vel,
            offsets,
        ),
        dim=-1,
    )


@pytest.mark.parametrize("local_frame", [True, False])
def test_default_selection_is_the_layout_checkpoints_were_trained_on(local_frame):
    s, commands = state(), torch.randn(NUM_ENVS, len(CommandIndex))
    mdp = ComposedMDP([], DEFAULT_OBS, DEFAULT_OBS, "cpu")

    policy, style = mdp.observations(s, commands, local_frame)

    expected = compute_proprioceptive_obs(s, local_frame)
    assert torch.equal(style, expected)
    assert torch.equal(policy, torch.cat((expected, commands), dim=-1))
    assert torch.equal(mdp.style_obs(s, local_frame), expected)
    assert observation_dim(DEFAULT_OBS) == 81 == expected.shape[-1]


def test_selection_composes_in_the_order_given_with_commands_last():
    s, commands = state(), torch.randn(NUM_ENVS, len(CommandIndex))
    mdp = ComposedMDP([], ("root_height", "dof_vel"), ("dof_pos",), "cpu")

    policy, style = mdp.observations(s, commands, True)

    assert torch.equal(policy, torch.cat((s.root_positions[:, 2:3], s.dof_velocities, commands), dim=-1))
    assert torch.equal(style, s.dof_positions)
    assert observation_dim(("root_height", "dof_vel")) == 29


def test_declared_dims_match_what_terms_produce():
    s = state()
    for name, term in OBSERVATIONS.items():
        assert term.fn(s, True).shape == (NUM_ENVS, term.dim), name


def test_only_declared_rewards_run_and_weight_zero_still_logs(specs):
    mdp = ComposedMDP(
        specs({"a": {"rewards": {"joint_vel": 2.0, "joint_acc": {"weight": 0.0}}}, "b": {"rewards": {"action_rate_l2": 1.0}}}),
        DEFAULT_OBS,
        DEFAULT_OBS,
        "cpu",
    )
    motion_ids = torch.tensor([0, 1, 0, 1, 0, 1])
    c = context()  # raises if anything asks for the reference motion

    total, raw, weighted = mdp.rewards(c, motion_ids)

    assert list(raw) == ["action_rate_l2", "joint_acc", "joint_vel"]  # registry order, not manifest order
    assert list(weighted) == ["action_rate_l2", "joint_acc", "joint_vel", "task"]
    assert (weighted["joint_acc"] == 0).all() and (raw["joint_acc"] != 0).all()
    assert torch.equal(weighted["joint_vel"][motion_ids == 1], torch.zeros(3))  # motion b didn't declare it
    assert torch.equal(weighted["action_rate_l2"][motion_ids == 0], torch.zeros(3))
    assert torch.allclose(total, weighted["action_rate_l2"] + weighted["joint_vel"])


def test_per_motion_kwargs_and_defaults_reach_the_term(specs):
    mdp = ComposedMDP(
        specs({"a": {"rewards": {"joint_vel": {"weight": 1.0, "scale": 7.0}}}, "b": {"rewards": {"joint_vel": 1.0}}, "c": {}}),
        DEFAULT_OBS,
        DEFAULT_OBS,
        "cpu",
    )
    assert mdp.reward_kwargs["joint_vel"]["scale"].tolist() == pytest.approx([7.0, REWARDS["joint_vel"].defaults["scale"]] * 1 + [1.0e-2])
    assert mdp.reward_weights.tolist() == [[1.0], [1.0], [0.0]]

    c = context()
    _, raw, _ = mdp.rewards(c, torch.tensor([0, 0, 0, 1, 1, 2]))
    error = torch.sum(c.state.dof_velocities**2, dim=-1)
    assert torch.equal(raw["joint_vel"][0], torch.exp(-torch.tensor(7.0) * error[0]))


def test_nothing_declared_is_a_valid_style_only_run(specs):
    mdp = ComposedMDP(specs({"a": {"termination": []}}), DEFAULT_OBS, DEFAULT_OBS, "cpu")
    motion_ids = torch.zeros(NUM_ENVS, dtype=torch.long)

    total, raw, weighted = mdp.rewards(context(), motion_ids)

    assert torch.equal(total, torch.zeros(NUM_ENVS))
    assert raw == {} and list(weighted) == ["task"]
    assert not mdp.died(context(), motion_ids).any()


def test_terminations_are_enabled_per_motion_and_tracking_is_lazy(specs):
    mdp = ComposedMDP(specs({"a": {}, "b": {"termination": []}}), DEFAULT_OBS, DEFAULT_OBS, "cpu")
    c = context()  # the default termination is unhealthy, which must not sample the reference
    c.state = c.state._replace(root_positions=torch.zeros(NUM_ENVS, 3))

    died = mdp.died(c, torch.tensor([0, 0, 0, 1, 1, 1]))

    assert mdp.termination_names == ["unhealthy"]
    assert died.tolist() == [True, True, True, False, False, False]


def test_progress_keeps_its_own_memory_across_steps(specs):
    mdp = ComposedMDP(specs({"a": {"rewards": {"progress": 1.0}}}), DEFAULT_OBS, DEFAULT_OBS, "cpu")
    motion_ids = torch.zeros(NUM_ENVS, dtype=torch.long)

    first, second = context(), context()
    _, raw_first, _ = mdp.rewards(first, motion_ids)
    _, raw_second, _ = mdp.rewards(second, motion_ids)

    distance = torch.norm(first.commanded_goal - first.root_positions_xy, dim=-1)
    next_distance = torch.norm(second.commanded_goal - second.root_positions_xy, dim=-1)
    assert torch.equal(raw_first["progress"], torch.zeros(NUM_ENVS))  # no previous step, so no progress yet
    assert torch.equal(raw_second["progress"], distance - next_distance)


@pytest.mark.parametrize(
    "motion, message",
    [
        ({"termination": "unhealhty"}, "unhealhty"),
        ({"termination": ["unhealthy", "unhealhty"]}, "unhealhty"),
        ({"rewards": {"lin_vel_trackin": 1.0}}, "lin_vel_trackin"),
        ({"rewards": {"lin_vel_tracking": {"weight": 1.0, "scael": 3.0}}}, "scael"),
    ],
)
def test_unknown_term_names_are_rejected_when_composing(specs, motion, message):
    with pytest.raises(ValueError, match=message):
        ComposedMDP(specs({"m": motion}), DEFAULT_OBS, DEFAULT_OBS, "cpu")


def test_shipped_manifest_composes():
    specs = load_manifest(str(REPO_ROOT / "configs/manifest.yaml"))
    joint_names = np.load(REPO_ROOT / "data/walk.npz")["dof_names"].tolist()  # the robot's joints
    assert ComposedMDP(specs, DEFAULT_OBS, DEFAULT_OBS, "cpu", joint_names=joint_names).reward_names


def test_unknown_observation_is_rejected():
    with pytest.raises(ValueError, match="dof_poz"):
        ComposedMDP([], ("dof_poz",), DEFAULT_OBS, "cpu")
    with pytest.raises(ValueError, match="dof_poz"):
        observation_dim(("dof_poz",))


def test_registries_hold_every_term_the_old_modules_named():
    assert list(REWARDS) == [
        "motion_tracking",
        "lin_vel_tracking",
        "yaw_vel_tracking",
        "target_hit",
        "line_following",
        "action_rate_l2",
        "joint_acc",
        "joint_vel",
        "progress",
        "root_lin_vel_l2",
        "root_ang_vel_l2",
        "upright",
        "support_centered",
        "standing_pose",
    ]
    assert {name: term.defaults for name, term in REWARDS.items()} == {
        "motion_tracking": {"dof_scale": 2.0, "key_body_scale": 5.0},
        "lin_vel_tracking": {"scale": 2.0},
        "yaw_vel_tracking": {"scale": 2.0},
        "target_hit": {"scale": 0.5},
        "line_following": {"scale": 2.0},
        "action_rate_l2": {"scale": 1.0},
        "joint_acc": {"scale": 1.0e-4},
        "joint_vel": {"scale": 1.0e-2},
        "progress": {"scale": 1.0},
        "root_lin_vel_l2": {"scale": 4.0},
        "root_ang_vel_l2": {"scale": 2.0},
        "upright": {"scale": 10.0},
        "support_centered": {"scale": 20.0},
        "standing_pose": {"scale": 2.0, "tolerance": 0.05},
    }
    assert list(TERMINATIONS) == ["deviation_from_motion", "unhealthy"]
