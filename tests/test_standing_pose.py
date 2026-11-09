"""The standing pose: the manifest's `standing.pose`, its per-joint target table, the
`standing_pose` reward's tolerance and its per-env standing gate."""

import math

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from conftest import REPO_ROOT  # noqa: E402
from moveitmoveit.manifest import COMMAND_DIM, load_manifest  # noqa: E402
from moveitmoveit.mdp import BodyState, ComposedMDP, StepContext  # noqa: E402

OBS = ("root_height",)
# Humanoid28's joints as the clips name them, in a different order than the robot reports them
CLIP_JOINTS = np.load(REPO_ROOT / "data/walk.npz")["dof_names"].tolist()
ROBOT_JOINTS = CLIP_JOINTS[::2] + CLIP_JOINTS[1::2]


@pytest.fixture
def specs(write_manifest):
    def _specs(motions: dict):
        return load_manifest(write_manifest({name: {"file": f"{name}.npz", **motion} for name, motion in motions.items()}))

    return _specs


def context(dof_positions: torch.Tensor) -> StepContext:
    n, d = dof_positions.shape
    state = BodyState(
        dof_positions,
        torch.zeros(n, d),
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n, 4, 3),
    )
    return StepContext(state, torch.zeros(n, d), torch.zeros(n, COMMAND_DIM), torch.zeros(n, d), torch.zeros(n, d), lambda: None)


POSE = {"right_hip_x": -0.12, "left_hip_x": 0.12, "right_elbow": -0.2}


@pytest.fixture
def mdp(specs) -> ComposedMDP:
    return ComposedMDP(
        specs(
            {
                "walk": {
                    "rewards": {"joint_vel": 1.0},
                    "standing": {"pose": POSE, "rewards": {"standing_pose": {"weight": 2.0, "scale": 3.0, "tolerance": 0.05}}},
                },
                "plain": {"rewards": {"joint_vel": 1.0}},
            }
        ),
        OBS,
        OBS,
        "cpu",
        joint_names=ROBOT_JOINTS,
    )


def target(mdp: ComposedMDP, motion_id: int = 0) -> torch.Tensor:
    return mdp.standing_pose_targets[motion_id]


# --- manifest and target table ----------------------------------------------------------------


def test_pose_is_parsed_and_defaults_to_empty(specs):
    walk, bare = specs({"walk": {"standing": {"pose": {"right_elbow": -1, "left_hip_x": 0.5}}}, "bare": {"standing": None}})

    assert walk.standing.pose == {"right_elbow": -1.0, "left_hip_x": 0.5}
    assert bare.standing.pose == {}


@pytest.mark.parametrize("pose, message", [({"right_elbow": "bent"}, "standing.pose.right_elbow"), ([0.1], "standing.pose")])
def test_invalid_poses_are_rejected(specs, pose, message):
    with pytest.raises(ValueError, match=message):
        specs({"m": {"standing": {"pose": pose}}})


def test_targets_follow_the_robot_joint_order_and_unlisted_joints_target_zero(mdp):
    assert mdp.standing_pose_targets.shape == (2, 28)
    for joint, value in POSE.items():
        assert target(mdp)[ROBOT_JOINTS.index(joint)] == pytest.approx(value)
    assert torch.count_nonzero(target(mdp)) == len(POSE)
    assert torch.all(target(mdp, 1) == 0.0)  # no standing block
    assert any(target(mdp)[CLIP_JOINTS.index(joint)] != value for joint, value in POSE.items())  # not clip order


def test_unknown_joints_terms_outside_standing_and_missing_joint_names_are_rejected(specs):
    with pytest.raises(ValueError, match="standing.pose"):
        ComposedMDP(specs({"m": {"standing": {"pose": {"right_hip_xx": 0.1}}}}), OBS, OBS, "cpu", joint_names=ROBOT_JOINTS)
    with pytest.raises(ValueError, match="standing-mode term"):
        ComposedMDP(specs({"m": {"rewards": {"standing_pose": 1.0}}}), OBS, OBS, "cpu", joint_names=ROBOT_JOINTS)
    with pytest.raises(ValueError, match="joint_names"):
        ComposedMDP(specs({"m": {"standing": {"rewards": {"standing_pose": 1.0}}}}), OBS, OBS, "cpu")


def test_without_joint_names_or_a_pose_term_nothing_changes(specs):
    plain = specs({"m": {"rewards": {"joint_vel": 1.0}, "standing": {"rewards": {"upright": 1.0}}}})

    assert ComposedMDP(plain, OBS, OBS, "cpu").standing_pose_targets is None
    assert "standing_pose" not in ComposedMDP(plain, OBS, OBS, "cpu", joint_names=ROBOT_JOINTS).reward_names


# --- the reward: target, perturbation, tolerance ----------------------------------------------


def standing_pose_raw(mdp: ComposedMDP, dof_positions: torch.Tensor, motion_ids=None) -> torch.Tensor:
    n = dof_positions.shape[0]
    motion_ids = torch.zeros(n, dtype=torch.long) if motion_ids is None else motion_ids
    _, raw, _ = mdp.rewards(context(dof_positions), motion_ids, torch.ones(n, dtype=torch.bool))
    return raw["standing_pose"]


def test_the_target_scores_one_and_deviations_within_tolerance_are_free(mdp):
    torch.manual_seed(0)
    at_target = target(mdp).repeat(3, 1)
    within = at_target + (torch.rand(3, 28) * 2.0 - 1.0) * 0.049  # every joint, up to the tolerance

    assert torch.allclose(standing_pose_raw(mdp, torch.cat((at_target, within))), torch.ones(6))


def test_excess_beyond_tolerance_is_squared_summed_and_scaled(mdp):
    elbow, hip = ROBOT_JOINTS.index("right_elbow"), ROBOT_JOINTS.index("left_hip_x")
    one_joint = target(mdp).clone()
    one_joint[elbow] -= 0.25
    two_joints = one_joint.clone()
    two_joints[hip] += 0.15

    raw = standing_pose_raw(mdp, torch.stack((one_joint, two_joints)))

    assert raw[0] == pytest.approx(math.exp(-3.0 * 0.2**2))  # 0.25 off, 0.05 of it tolerated
    assert raw[1] == pytest.approx(math.exp(-3.0 * (0.2**2 + 0.1**2)))
    assert raw[1] < raw[0] < 1.0


def test_a_wide_arms_out_stance_scores_far_below_the_target(mdp):
    wide = target(mdp).clone()
    for joint, value in {"right_shoulder_x": -1.4, "left_shoulder_x": 1.4, "right_hip_x": -0.35, "left_hip_x": 0.35}.items():
        wide[ROBOT_JOINTS.index(joint)] = value

    assert standing_pose_raw(mdp, wide.unsqueeze(0)).item() < 0.01


# --- per-env standing gate --------------------------------------------------------------------


def test_only_standing_envs_are_weighted_by_the_pose(mdp):
    motion_ids = torch.tensor([0, 0, 1, 1])
    standing = torch.tensor([True, False, True, False])
    off_target = target(mdp).repeat(4, 1)
    off_target[:, ROBOT_JOINTS.index("right_elbow")] += 0.3

    _, raw, weighted = mdp.rewards(context(off_target), motion_ids, standing)

    assert weighted["standing_pose"][0] == pytest.approx(2.0 * math.exp(-3.0 * 0.25**2))
    assert weighted["standing_pose"][1:].tolist() == [0.0, 0.0, 0.0]  # walking, or no standing mode
    assert weighted["joint_vel"][0] == 0.0 and weighted["joint_vel"][1] == 1.0
    _, _, unmasked = mdp.rewards(context(off_target), motion_ids)
    assert torch.all(unmasked["standing_pose"] == 0.0)


# --- the shipped target -----------------------------------------------------------------------


MIRRORED_SIGN = {"x": -1.0, "y": 1.0, "z": -1.0}  # mirroring left/right flips rolls and yaws


@pytest.mark.parametrize("manifest", ["manifest.yaml", "goal_manifest.yaml"])
def test_shipped_manifests_ask_for_a_symmetric_pose_on_humanoid28_joints(manifest):
    (spec,) = load_manifest(str(REPO_ROOT / "configs" / manifest))
    mdp = ComposedMDP([spec], OBS, OBS, "cpu", joint_names=ROBOT_JOINTS)

    pose = dict(zip(ROBOT_JOINTS, target(mdp).tolist()))
    assert "standing_pose" in mdp.reward_names and spec.standing.pose
    for joint, value in pose.items():
        if joint.startswith("right_"):
            mirror = pose[joint.replace("right_", "left_")]
            sign = MIRRORED_SIGN.get(joint[-1], 1.0) if joint[-2] == "_" else 1.0
            assert mirror == pytest.approx(sign * value), joint
    # feet apart and flat: each hip roll undone at its ankle; arms down: shoulders open well under 45°
    assert pose["right_hip_x"] < 0.0 and pose["right_ankle_x"] == pytest.approx(-pose["right_hip_x"])
    assert -0.785 < pose["right_shoulder_x"] < 0.0
