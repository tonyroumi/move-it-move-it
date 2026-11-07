"""Characterization of the per-motion tables compiled from a manifest."""

import ast

import pytest
import torch

from conftest import REPO_ROOT
from moveitmoveit.commands import COMMAND_DIM, CommandIndex
from moveitmoveit.motion_set import MotionSet
from moveitmoveit.terms import NUM_REWARDS, REWARD_KWARG_DEFAULTS, RewardIndex, TerminationIndex

# The multi-motion manifest from commit e1622a8, inlined (its files become synthetic clips).
HISTORICAL_MOTIONS = {
    "standing_still": {
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


@pytest.fixture
def compile_motions(write_manifest, write_clip):
    """Compile motions from a manifest, each with its own synthetic clip."""

    def _compile(motions: dict) -> MotionSet:
        with_clips = {name: {**motion, "file": write_clip(name, 5 + i)} for i, (name, motion) in enumerate(motions.items())}
        return MotionSet.from_manifest(write_manifest(with_clips), "cpu")

    return _compile


def test_motion_ids_follow_manifest_order(compile_motions):
    motions = compile_motions({"zebra": {}, "apple": {}, "mango": {}})

    assert motions.names == ["zebra", "apple", "mango"]
    assert motions.name_to_id == {"zebra": 0, "apple": 1, "mango": 2}
    assert motions.num_motions == 3


def test_motion_ids_index_the_same_clips_in_the_library(compile_motions):
    # each synthetic clip has a distinct length, so this catches any table/clip misalignment
    motions = compile_motions({"zebra": {}, "apple": {}, "mango": {}})

    assert motions.library.num_motions == motions.num_motions
    assert motions.library.num_frames.tolist() == [5, 6, 7]


def test_historical_manifest_tables(compile_motions):
    motions = compile_motions(HISTORICAL_MOTIONS)

    assert motions.sample_prob.tolist() == pytest.approx([0.2 / 1.2, 1.0 / 1.2])
    assert motions.command_zero_prob.tolist() == pytest.approx([1.0, 0.15])

    assert motions.command_low[1].tolist() == pytest.approx([-1.5, -1.5, -2.0, 0.0, 0.0, 0.0, 0.0])
    assert motions.command_high[1].tolist() == pytest.approx([1.5, 1.5, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert motions.command_low[0].tolist() == [0.0] * COMMAND_DIM

    # columns: motion_tracking lin_vel yaw_vel target_hit line_following action_rate joint_acc joint_vel progress
    assert motions.reward_weights[0].tolist() == pytest.approx([0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.0])
    assert motions.reward_weights[1].tolist() == pytest.approx([0.0, 3.0, 1.0, 0.0, 1.0, 0.5, 0.5, 0.3, 0.0])

    assert motions.reward_kwargs["action_rate_l2"]["scale"].tolist() == [2.0, 2.0]
    assert motions.termination_flags.tolist() == [[False, True], [False, True]]


def test_command_scalars_ranges_and_absent_keys(compile_motions):
    motions = compile_motions(
        {"m": {"commands": {"lin_x": [1.0, 2.0], "binary_key_0": 1.0, "goal_y": [-3.0, 3.0]}}},
    )

    low, high = motions.command_low[0], motions.command_high[0]
    assert low[CommandIndex.LIN_X] == 1.0 and high[CommandIndex.LIN_X] == 2.0
    assert low[CommandIndex.BINARY_KEY_0] == 1.0 and high[CommandIndex.BINARY_KEY_0] == 1.0
    assert low[CommandIndex.GOAL_Y] == -3.0 and high[CommandIndex.GOAL_Y] == 3.0
    assert low[CommandIndex.YAW] == 0.0 and high[CommandIndex.YAW] == 0.0
    assert motions.command_zero_prob.tolist() == [0.0]


def test_reward_spec_forms_and_kwarg_defaults(compile_motions):
    motions = compile_motions(
        {
            "m": {
                "rewards": {
                    "lin_vel_tracking": 2.5,
                    "yaw_vel_tracking": {"scale": 7.0},
                    "joint_vel": {"weight": 0.25},
                    "motion_tracking": {"weight": 1.5, "key_body_scale": 9.0},
                },
            },
            "no_rewards": {},
        },
    )

    weights = motions.reward_weights
    assert weights[0, RewardIndex.LIN_VEL_TRACKING] == 2.5
    assert weights[0, RewardIndex.YAW_VEL_TRACKING] == 1.0  # dict without weight gets 1.0
    assert weights[0, RewardIndex.JOINT_VEL] == 0.25
    assert weights[0, RewardIndex.MOTION_TRACKING] == 1.5
    assert weights[0, RewardIndex.PROGRESS] == 0.0  # omitted reward has weight 0
    assert weights[1].tolist() == [0.0] * NUM_REWARDS

    assert motions.reward_kwargs["yaw_vel_tracking"]["scale"].tolist() == [7.0, 2.0]
    assert motions.reward_kwargs["motion_tracking"]["key_body_scale"].tolist() == [9.0, 5.0]
    assert motions.reward_kwargs["motion_tracking"]["dof_scale"].tolist() == [2.0, 2.0]

    ids = torch.tensor([1, 0, 1])
    assert motions.reward_weights_for(ids)[:, RewardIndex.LIN_VEL_TRACKING].tolist() == [0.0, 2.5, 0.0]
    assert motions.reward_kwargs_for("yaw_vel_tracking", ids)["scale"].tolist() == [2.0, 7.0, 2.0]

    # every reward has kwargs, ready to splat into the reward function
    assert set(motions.reward_kwargs) == set(REWARD_KWARG_DEFAULTS)


def test_termination_forms_default_to_unhealthy(compile_motions):
    motions = compile_motions(
        {
            "default": {},
            "string": {"termination": "deviation_from_motion"},
            "list": {"termination": ["deviation_from_motion", "unhealthy"]},
            "none": {"termination": []},
        },
    )

    deviation, unhealthy = TerminationIndex.DEVIATION_FROM_MOTION, TerminationIndex.UNHEALTHY
    flags = motions.termination_flags
    assert flags[0, deviation] == False and flags[0, unhealthy] == True  # noqa: E712
    assert flags[1, deviation] == True and flags[1, unhealthy] == False  # noqa: E712
    assert flags[2, deviation] == True and flags[2, unhealthy] == True  # noqa: E712
    assert flags[3, deviation] == False and flags[3, unhealthy] == False  # noqa: E712
    assert motions.termination_flags_for(torch.tensor([1]))[0, deviation] == True  # noqa: E712


def test_motion_sampling_is_weighted_and_seed_stable(compile_motions):
    motions = compile_motions(
        {
            "a": {"sample_prob": 0.2},
            "b": {"sample_prob": 1.0},
            "c": {"sample_prob": 0.8},
            "never": {"sample_prob": 0},
        },
    )
    assert motions.sample_prob.tolist() == pytest.approx([0.1, 0.5, 0.4, 0.0])

    torch.manual_seed(0)
    sampled = motions.sample_ids(50_000)
    torch.manual_seed(0)
    expected = torch.multinomial(motions.sample_prob, 50_000, replacement=True)
    assert torch.equal(sampled, expected)

    frequencies = torch.bincount(sampled, minlength=4) / 50_000
    assert frequencies.tolist() == pytest.approx([0.1, 0.5, 0.4, 0.0], abs=0.01)


def test_sample_commands_draw_order_and_full_vector_zeroing(compile_motions):
    motions = compile_motions(
        {
            "m": {
                "commands": {"lin_x": [1.0, 2.0], "binary_key_0": 1.0, "binary_key_1": [0.0, 1.0]},
                "zero_command_prob": 0.5,
            },
        },
    )
    ids = torch.zeros(10_000, dtype=torch.long)

    torch.manual_seed(3)
    commands = motions.sample_commands(ids)

    torch.manual_seed(3)
    low, high = motions.command_low[ids], motions.command_high[ids]
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
