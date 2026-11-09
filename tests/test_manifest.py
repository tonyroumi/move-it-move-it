"""The manifest parser: valid forms, strict rejection, and its torch-free guarantee."""

import ast
from dataclasses import replace

import pytest

from conftest import REPO_ROOT
from moveitmoveit.manifest import RewardSpec, load_manifest


def test_parses_every_declared_form(write_manifest):
    specs = load_manifest(
        write_manifest(
            {
                "full": {
                    "file": "full.npz",
                    "sample_prob": 0.5,
                    "commands": {"lin_x": [0.0, 1.0], "binary_key_0": 1},
                    "zero_command_prob": 0.25,
                    "rewards": {"lin_vel_tracking": 3, "joint_vel": {"weight": 0.5, "scale": 1.0e-3}, "yaw_vel_tracking": {}},
                    "termination": ["unhealthy", "deviation_from_motion"],
                },
                "defaults": {"file": "defaults.npz"},
                "empty_sections": {"file": "empty.npz", "commands": None, "rewards": None},
            }
        )
    )

    assert [spec.name for spec in specs] == ["full", "defaults", "empty_sections"]

    full = specs[0]
    assert full.file == "full.npz"
    assert full.sample_prob == 0.5
    assert full.commands == {"lin_x": (0.0, 1.0), "binary_key_0": (1.0, 1.0)}
    assert full.zero_command_prob == 0.25
    assert full.rewards == {
        "lin_vel_tracking": RewardSpec(3.0, {}),
        "joint_vel": RewardSpec(0.5, {"scale": 1.0e-3}),
        "yaw_vel_tracking": RewardSpec(1.0, {}),
    }
    assert full.terminations == ("unhealthy", "deviation_from_motion")

    defaults = specs[1]
    assert (defaults.sample_prob, defaults.zero_command_prob) == (1.0, 0.0)
    assert (defaults.commands, defaults.rewards, defaults.terminations) == ({}, {}, ("unhealthy",))
    assert specs[2].commands == {} and specs[2].rewards == {}


def test_single_termination_string_and_empty_list(write_manifest):
    specs = load_manifest(
        write_manifest(
            {
                "string": {"file": "a.npz", "termination": "deviation_from_motion"},
                "none": {"file": "b.npz", "termination": []},
            }
        )
    )
    assert specs[0].terminations == ("deviation_from_motion",)
    assert specs[1].terminations == ()


def test_a_zero_weight_motion_is_allowed_while_another_can_be_sampled(write_manifest):
    specs = load_manifest(write_manifest({"on": {"file": "a.npz"}, "off": {"file": "b.npz", "sample_prob": 0}}))
    assert [spec.sample_prob for spec in specs] == [1.0, 0.0]


def test_shipped_manifest_is_valid():
    assert load_manifest(str(REPO_ROOT / "src/configs/manifest.yaml"))


def test_smooth_walk_manifest_is_the_baseline_plus_walking_smoothness_rewards():
    (baseline,) = load_manifest(str(REPO_ROOT / "src/configs/manifest.yaml"))
    (smooth,) = load_manifest(str(REPO_ROOT / "src/configs/smooth_walk_manifest.yaml"))

    added = {name: spec for name, spec in smooth.rewards.items() if name not in baseline.rewards}
    assert set(added) == {"action_rate_l2", "joint_acc"}
    assert all(spec.weight > 0.0 and set(spec.kwargs) == {"scale"} for spec in added.values())
    assert replace(smooth, rewards={name: smooth.rewards[name] for name in baseline.rewards}) == baseline


@pytest.mark.parametrize(
    "motion, message",
    [
        ({"commands": {"lin_xx": [0.0, 1.0]}}, "lin_xx"),
        ({"zero_command_probability": 1.0}, "zero_command_probability"),
        ({"termination": 3}, "termination"),
        ({"termination": [3]}, "termination"),
        ({"rewards": {"lin_vel_tracking": "high"}}, "lin_vel_tracking"),
        ({"rewards": {"joint_acc": {"scale": "1e-4"}}}, "joint_acc.scale"),
        ({"commands": {"lin_x": [1.0, 0.0]}}, "reversed"),
        ({"commands": {"lin_x": [0.0, 1.0, 2.0]}}, "lin_x"),
        ({"commands": {"lin_x": "fast"}}, "lin_x"),
        ({"commands": ["lin_x"]}, "commands"),
        ({"sample_prob": -1.0}, "sample_prob"),
        ({"zero_command_prob": 1.5}, "zero_command_prob"),
    ],
)
def test_invalid_motions_are_rejected(write_manifest, motion, message):
    with pytest.raises(ValueError, match=message):
        load_manifest(write_manifest({"m": {"file": "m.npz", **motion}}))


def test_a_motion_needs_a_file(write_manifest):
    with pytest.raises(ValueError, match="file"):
        load_manifest(write_manifest({"m": {"sample_prob": 1.0}}))


def test_rejects_malformed_documents(write_manifest):
    with pytest.raises(ValueError, match="no motions"):
        load_manifest(write_manifest({}))
    with pytest.raises(ValueError, match="top-level"):
        load_manifest(write_manifest({"m": {"file": "m.npz"}}, typo=1))
    with pytest.raises(ValueError, match="sample_prob"):
        load_manifest(write_manifest({"m": {"file": "m.npz", "sample_prob": 0}}))


def test_manifest_layer_does_not_import_torch():
    tree = ast.parse((REPO_ROOT / "src/moveitmoveit/manifest.py").read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(("." * node.level) + (node.module or ""))

    assert imported <= {"dataclasses", "enum", "typing", "yaml"}
