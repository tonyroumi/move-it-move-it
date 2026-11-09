"""scripts/view_standing_pose.py's pure parts: which pose it shows, in which joint order, and how
high it puts the root. The viewer itself needs Isaac Sim and isn't run here."""

import importlib.util

import numpy as np
import pytest

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from conftest import REPO_ROOT  # noqa: E402
from moveitmoveit.manifest import load_manifest  # noqa: E402
from moveitmoveit.mdp import ComposedMDP  # noqa: E402

OBS = ("root_height",)
# Humanoid28's joints as the clips name them, in a different order than the robot reports them
CLIP_JOINTS = np.load(REPO_ROOT / "src/data/walk.npz")["dof_names"].tolist()
ROBOT_JOINTS = CLIP_JOINTS[::2] + CLIP_JOINTS[1::2]

_spec = importlib.util.spec_from_file_location("view_standing_pose", REPO_ROOT / "scripts/view_standing_pose.py")
view = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(view)


def composed(specs) -> ComposedMDP:
    return ComposedMDP(specs, OBS, OBS, "cpu", joint_names=ROBOT_JOINTS)


@pytest.mark.parametrize("manifest", ["manifest.yaml", "goal_manifest.yaml"])
def test_shows_the_shipped_pose_in_the_robot_joint_order(manifest):
    specs = load_manifest(str(REPO_ROOT / "src/configs" / manifest))

    motion_id, target = view.standing_pose(specs, composed(specs).standing_pose_targets)

    pose = specs[motion_id].standing.pose
    assert pose and target.shape == (len(ROBOT_JOINTS),)
    for joint, value in zip(ROBOT_JOINTS, target.tolist()):
        assert value == pytest.approx(pose.get(joint, 0.0)), joint


def test_picks_the_named_motion_or_the_first_with_a_pose(write_manifest):
    specs = load_manifest(
        write_manifest(
            {
                "plain": {"file": "plain.npz"},
                "walk": {"file": "walk.npz", "standing": {"pose": {"right_elbow": -0.2}}},
                "run": {"file": "run.npz", "standing": {"pose": {"left_knee": 0.3}}},
            }
        )
    )
    targets = composed(specs).standing_pose_targets

    assert view.standing_pose(specs, targets)[0] == 1
    motion_id, target = view.standing_pose(specs, targets, "run")
    assert motion_id == 2 and target[ROBOT_JOINTS.index("left_knee")] == pytest.approx(0.3)
    with pytest.raises(ValueError, match="'plain' declares no standing.pose"):
        view.standing_pose(specs, targets, "plain")
    with pytest.raises(ValueError, match="'jump' declares no standing.pose"):
        view.standing_pose(specs, targets, "jump")


def test_a_manifest_without_any_pose_is_rejected(write_manifest):
    specs = load_manifest(write_manifest({"plain": {"file": "plain.npz"}}))

    with pytest.raises(ValueError, match="No motion"):
        view.standing_pose(specs, composed(specs).standing_pose_targets)


def test_root_drops_so_the_lower_sole_rests_on_the_ground():
    # feet at 0.30/0.28 with the root at 1.12: the lower sole (0.28 - 0.05) is 0.23 above the ground
    assert view.grounded_root_height(1.12, [0.30, 0.28]) == pytest.approx(1.12 - 0.23)
    assert view.grounded_root_height(1.0, [view.FOOT_SOLE_DEPTH] * 2) == pytest.approx(1.0)
    assert view.grounded_root_height(1.0, [0.05, 0.05], ground_z=0.1) == pytest.approx(1.1)
