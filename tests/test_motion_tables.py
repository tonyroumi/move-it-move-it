"""Characterization of the per-motion tables compiled from a manifest."""

import ast

import pytest
import torch

from conftest import REPO_ROOT
from moveitmoveit.commands import COMMAND_DIM, CommandIndex
from moveitmoveit.motion_manager import MotionManager
from moveitmoveit.rewards import NUM_REWARDS, REWARD_KWARG_DEFAULTS, RewardIndex
from moveitmoveit.terminations import TerminationIndex

# The multi-motion manifest from commit e1622a8, inlined (minus its file paths).
HISTORICAL_MOTIONS = {
    "standing_still": {
        "file": "standing_still.npz",
        "sample_prob": 0.2,
        "commands": {"lin_x": [0.0, 0.0], "lin_y": [0.0, 0.0], "yaw": [0.0, 0.0], "binary_key_0": 0.0},
        "zero_command_prob": 1.0,
        "rewards": {
            "lin_vel_tracking": {"weight": 0.0, "scale": 2.0},
            "yaw_vel_tracking": {"weight": 0.0, "scale": 2.0},
            "line_following": {"weight": 0.0, "scale": 2.0},
            "action_rate_l2": {"weight": 0.5, "scale": 2.0},
            "joint_acc": {"weight": 0.5, "scale": 1.0e-4},
            "joint_vel": {"weight": 0.5, "scale": 1.0e-2},
        },
        "termination": "unhealthy",
    },
    "walk": {
        "file": "walk.npz",
        "sample_prob": 1.0,
        "commands": {"lin_x": [-1.5, 1.5], "lin_y": [-1.5, 1.5], "yaw": [-2.0, 2.0], "binary_key_0": 0.0},
        "zero_command_prob": 0.15,
        "rewards": {
            "lin_vel_tracking": {"weight": 3.0, "scale": 2.0},
            "yaw_vel_tracking": {"weight": 1.0, "scale": 2.0},
            "line_following": {"weight": 1.0, "scale": 2.0},
            "action_rate_l2": {"weight": 0.5, "scale": 2.0},
            "joint_acc": {"weight": 0.5, "scale": 1.0e-4},
            "joint_vel": {"weight": 0.3, "scale": 1.0e-2},
        },
        "termination": "unhealthy",
    },
}


def compile_motions(write_manifest, motions: dict) -> MotionManager:
    return MotionManager(write_manifest(motions), "cpu")


def test_motion_ids_follow_manifest_order(write_manifest):
    motions = {name: {"file": f"{name}.npz"} for name in ("zebra", "apple", "mango")}
    manager = compile_motions(write_manifest, motions)

    assert manager.motion_names == ["zebra", "apple", "mango"]
    assert manager.motion_files == ["zebra.npz", "apple.npz", "mango.npz"]
    assert manager.motion_name_to_id == {"zebra": 0, "apple": 1, "mango": 2}
    assert manager.num_motions == 3


def test_historical_manifest_tables(write_manifest):
    manager = compile_motions(write_manifest, HISTORICAL_MOTIONS)

    assert manager.motion_sample_prob.tolist() == pytest.approx([0.2 / 1.2, 1.0 / 1.2])
    assert manager.command_zero_prob.tolist() == pytest.approx([1.0, 0.15])

    assert manager.command_low[1].tolist() == pytest.approx([-1.5, -1.5, -2.0, 0.0, 0.0, 0.0, 0.0])
    assert manager.command_high[1].tolist() == pytest.approx([1.5, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert manager.command_low[0].tolist() == [0.0] * COMMAND_DIM

    # columns: motion_tracking lin_vel yaw_vel target_hit line_following action_rate joint_acc joint_vel progress
    assert manager.reward_weights[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0])
    assert manager.reward_weights[1].tolist() == pytest.approx([0.0, 3.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.3, 0.0])

    assert manager.reward_kwargs["action_rate_l2"]["scale"].tolist() == [2.0, 2.0]
    assert manager.termination_flags.tolist() == [[False, True], [False, True]]


def test_command_scalars_ranges_and_absent_keys(write_manifest):
    manager = compile_motions(
        write_manifest,
        {"m": {"file": "m.npz", "commands": {"lin_x": [1.0, 2.0], "binary_key_0": 1.0, "goal_y": [-3.0, 3.0]}}},
    )

    low, high = manager.command_low[0], manager.command_high[0]
    assert low[CommandIndex.LIN_X] == 1.0 and high[CommandIndex.LIN_X] == 2.0
    assert low[CommandIndex.BINARY_KEY_0] == 1.0 and high[CommandIndex.BINARY_KEY_0] == 1.0
    assert low[CommandIndex.GOAL_Y] == -3.0 and high[CommandIndex.GOAL_Y] == 3.0
    assert low[CommandIndex.YAW] == 0.0 and high[CommandIndex.YAW] == 0.0
    assert manager.command_zero_prob.tolist() == [0.0]


def test_reward_spec_forms_and_kwarg_defaults(write_manifest):
    manager = compile_motions(
        write_manifest,
        {
            "m": {
                "file": "m.npz",
                "rewards": {
                    "lin_vel_tracking": 2.5,
                    "yaw_vel_tracking": {"scale": 7.0},
                    "joint_vel": {"weight": 0.25},
                    "motion_tracking": {"weight": 1.5, "key_body_scale": 9.0},
                },
            },
            "no_rewards": {"file": "n.npz"},
        },
    )

    weights = manager.reward_weights
    assert weights[0, RewardIndex.LIN_VEL_TRACKING] == 2.5
    assert weights[0, RewardIndex.YAW_VEL_TRACKING] == 1.0  # dict without weight gets 1.0
    assert weights[0, RewardIndex.JOINT_VEL] == 0.25
    assert weights[0, RewardIndex.MOTION_TRACKING] == 1.5
    assert weights[0, RewardIndex.PROGRESS] == 0.0  # omitted reward has weight 0
    assert weights[1].tolist() == [0.0] * NUM_REWARDS

    assert manager.reward_kwargs["yaw_vel_tracking"]["scale"].tolist() == [7.0, 2.0]
    assert manager.reward_kwargs["motion_tracking"]["key_body_scale"].tolist() == [9.0, 5.0]
    assert manager.reward_kwargs["motion_tracking"]["dof_scale"].tolist() == [2.0, 2.0]

    ids = torch.tensor([1, 0, 1])
    assert manager.reward_weights_for(ids)[:, RewardIndex.LIN_VEL_TRACKING].tolist() == [0.0, 2.5, 0.0]
    assert manager.reward_kwargs_for("yaw_vel_tracking", ids)["scale"].tolist() == [2.0, 7.0, 2.0]

    # every reward has kwargs, ready to splat into the reward function
    assert set(manager.reward_kwargs) == set(REWARD_KWARG_DEFAULTS)


def test_termination_forms_default_to_unhealthy(write_manifest):
    manager = compile_motions(
        write_manifest,
        {
            "default": {"file": "a.npz"},
            "string": {"file": "b.npz", "termination": "deviation_from_motion"},
            "list": {"file": "c.npz", "termination": ["deviation_from_motion", "unhealthy"]},
            "none": {"file": "d.npz", "termination": []},
        },
    )

    deviation, unhealthy = TerminationIndex.DEVIATION_FROM_MOTION, TerminationIndex.UNHEALTHY
    flags = manager.termination_flags
    assert flags[0, deviation] == False and flags[0, unhealthy] == True  # noqa: E712
    assert flags[1, deviation] == True and flags[1, unhealthy] == False  # noqa: E712
    assert flags[2, deviation] == True and flags[2, unhealthy] == True  # noqa: E712
    assert flags[3, deviation] == False and flags[3, unhealthy] == False  # noqa: E712
    assert manager.termination_flags_for(torch.tensor([1]))[0, deviation] == True  # noqa: E712


def test_motion_sampling_is_weighted_and_seed_stable(write_manifest):
    manager = compile_motions(
        write_manifest,
        {
            "a": {"file": "a.npz", "sample_prob": 0.2},
            "b": {"file": "b.npz", "sample_prob": 1.0},
            "c": {"file": "c.npz", "sample_prob": 0.8},
            "never": {"file": "d.npz", "sample_prob": 0},
        },
    )
    assert manager.motion_sample_prob.tolist() == pytest.approx([0.1, 0.5, 0.4, 0.0])

    torch.manual_seed(0)
    sampled = manager.sample_motion(50_000)
    torch.manual_seed(0)
    expected = torch.multinomial(manager.motion_sample_prob, 50_000, replacement=True)
    assert torch.equal(sampled, expected)

    frequencies = torch.bincount(sampled, minlength=4) / 50_000
    assert frequencies.tolist() == pytest.approx([0.1, 0.5, 0.4, 0.0], abs=0.01)


def test_sample_commands_draw_order_and_full_vector_zeroing(write_manifest):
    manager = compile_motions(
        write_manifest,
        {
            "m": {
                "file": "m.npz",
                "commands": {"lin_x": [1.0, 2.0], "binary_key_0": 1.0, "binary_key_1": [0.0, 1.0]},
                "zero_command_prob": 0.5,
            },
        },
    )
    ids = torch.zeros(10_000, dtype=torch.long)

    torch.manual_seed(3)
    commands = manager.sample_commands(ids)

    torch.manual_seed(3)
    low, high = manager.command_low[ids], manager.command_high[ids]
    expected = low + torch.rand_like(low) * (high - low)
    expected[torch.rand(ids.shape[0]) < 0.5] = 0.0
    assert torch.equal(commands, expected)

    # zeroing clears the entire vector, including a declared constant binary key
    zeroed = (commands == 0).all(dim=-1)
    assert zeroed.float().mean().item() == pytest.approx(0.5, abs=0.03)
    assert (commands[~zeroed, CommandIndex.BINARY_KEY_0] == 1.0).all()


def test_reward_identity_lists_stay_in_sync():
    """Weights multiply the env's stacked reward terms by position, so these orders must agree."""
    tree = ast.parse((REPO_ROOT / "src/moveitmoveit/env.py").read_text())
    env_terms = next(
        [key.value for key in node.value.keys]
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "reward_terms" for t in node.targets)
        and isinstance(node.value, ast.Dict)
    )
    enum_names = [reward.name.lower() for reward in RewardIndex]

    assert env_terms == enum_names
    assert list(REWARD_KWARG_DEFAULTS) == enum_names
