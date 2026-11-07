"""Characterization of the per-motion tables compiled from a manifest."""

import pytest
import torch

from moveitmoveit.manifest import COMMAND_DIM, CommandIndex
from moveitmoveit.motion_set import MotionSet

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
