"""Standing mode without Isaac Lab: the manifest's `standing` block and which commands select it."""

import pytest
import torch

from conftest import REPO_ROOT
from moveitmoveit.manifest import COMMAND_DIM, CommandIndex, RewardSpec, StandingSpec, load_manifest
from moveitmoveit.motion_set import MotionSet

GOAL = slice(CommandIndex.GOAL_X, CommandIndex.GOAL_Y + 1)


# --- manifest ---------------------------------------------------------------------------------


def test_standing_is_off_unless_declared_and_an_empty_block_takes_the_defaults(write_manifest):
    specs = load_manifest(
        write_manifest(
            {
                "absent": {"file": "a.npz"},
                "empty": {"file": "b.npz", "standing": None},
                "full": {
                    "file": "c.npz",
                    "standing": {
                        "lin_vel_threshold": 0.2,
                        "yaw_vel_threshold": 0.0,
                        "goal_radius": 1,
                        "rewards": {"root_lin_vel_l2": 2, "joint_vel": {"weight": 0.5, "scale": 0.05}},
                    },
                },
            }
        )
    )

    assert specs[0].standing is None
    assert specs[1].standing == StandingSpec(0.1, 0.1, 0.3, {})
    assert specs[2].standing == StandingSpec(
        0.2, 0.0, 1.0, {"root_lin_vel_l2": RewardSpec(2.0, {}), "joint_vel": RewardSpec(0.5, {"scale": 0.05})}
    )


@pytest.mark.parametrize(
    "standing, message",
    [
        ({"lin_vel_treshold": 0.1}, "lin_vel_treshold"),
        ({"goal_radius": -0.1}, "goal_radius"),
        ({"yaw_vel_threshold": "slow"}, "yaw_vel_threshold"),
        ({"rewards": {"joint_vel": {"weight": "high"}}}, "standing.rewards.joint_vel.weight"),
        ([0.1], "standing"),
    ],
)
def test_invalid_standing_blocks_are_rejected(write_manifest, standing, message):
    with pytest.raises(ValueError, match=message):
        load_manifest(write_manifest({"m": {"file": "m.npz", "standing": standing}}))


@pytest.mark.parametrize("manifest", ["manifest.yaml", "goal_manifest.yaml"])
def test_shipped_manifests_declare_standing(manifest):
    (spec,) = load_manifest(str(REPO_ROOT / "src/configs" / manifest))

    assert spec.standing is not None and spec.standing.rewards
    assert spec.zero_command_prob > 0.0  # resamples train both modes and the switches between them


# --- which commands stand ---------------------------------------------------------------------


@pytest.fixture
def motions(write_manifest, write_clip) -> MotionSet:
    return MotionSet.from_manifest(
        write_manifest(
            {
                "walk": {
                    "file": write_clip("a", 5),
                    "commands": {"lin_x": [0.5, 1.0], "yaw": [-1.0, 1.0]},
                    "zero_command_prob": 0.5,
                    "standing": {"lin_vel_threshold": 0.1, "yaw_vel_threshold": 0.2},
                },
                "to_goal": {
                    "file": write_clip("b", 6),
                    "commands": {"goal_x": [2.0, 3.0]},
                    "zero_command_prob": 0.5,
                    "standing": {"goal_radius": 0.3},
                },
                "plain": {"file": write_clip("c", 7), "commands": {"lin_x": [0.5, 1.0]}},
            }
        ),
        "cpu",
    )


def observed(lin_x=0.0, lin_y=0.0, yaw=0.0, goal=(0.0, 0.0), key=0.0) -> list[float]:
    """One env's command as the policy observes it: the goal slots hold its offset from the root."""
    row = [0.0] * COMMAND_DIM
    row[CommandIndex.LIN_X], row[CommandIndex.LIN_Y], row[CommandIndex.YAW] = lin_x, lin_y, yaw
    row[CommandIndex.BINARY_KEY_0] = key
    row[CommandIndex.GOAL_X], row[CommandIndex.GOAL_Y] = goal
    return row


def test_mixed_envs_get_their_own_mode(motions):
    cases = [
        (0, observed(), True),  # zero velocity command
        (0, observed(lin_x=0.8), False),
        (0, observed(lin_x=0.06, lin_y=0.08), True),  # norm 0.1: within the threshold
        (0, observed(lin_x=0.08, lin_y=0.08), False),  # norm 0.113
        (0, observed(yaw=0.5), False),  # turning in place is locomotion
        (0, observed(yaw=-0.2), True),
        (0, observed(key=1.0), True),  # key slots aren't motion
        (1, observed(goal=(0.0, 0.0)), True),  # a zeroed goal command: the goal is at the root
        (1, observed(goal=(2.5, 0.0)), False),  # velocity slots are 0, but there is a goal to walk to
        (1, observed(goal=(0.2, -0.2)), True),  # arrived within goal_radius
        (1, observed(goal=(-0.3, 0.1)), False),
        (2, observed(), False),  # no standing block: never standing
    ]
    motion_ids = torch.tensor([motion_id for motion_id, _, _ in cases])
    commands = torch.tensor([command for _, command, _ in cases])

    standing = motions.standing(motion_ids, commands)

    assert standing.tolist() == [expected for _, _, expected in cases]
    assert motions.can_stand.tolist() == [True, True, False]


def test_sampled_commands_stand_exactly_when_zeroed(motions):
    torch.manual_seed(0)
    motion_ids = torch.tensor([0] * 200 + [1] * 200)
    roots_xy = torch.randn(400, 2) * 10.0

    commands = motions.sample_commands(motion_ids, roots_xy)
    view = commands.clone()
    view[200:, GOAL] -= roots_xy[200:]  # what policy_commands makes of it, up to a heading rotation
    standing = motions.standing(motion_ids, view)

    zeroed = torch.cat((commands[:200, :3].abs().sum(-1) == 0, torch.all(view[200:, GOAL] == 0, dim=-1)))
    assert torch.equal(standing, zeroed)
    assert 0 < standing[:200].sum() < 200 and 0 < standing[200:].sum() < 200  # both modes get trained
