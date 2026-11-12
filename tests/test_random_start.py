"""A manifest motion's `random_start` picks where a reset starts its clip clock. No simulator."""

from pathlib import Path

import pytest
import torch

from moveitmoveit.manifest import load_manifest
from moveitmoveit.motion_set import MotionSet
from moveitmoveit.utils.paths import CONFIGS_DIR
from test_loop import write_clip, write_manifest

NUM_RESETS = 2000


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(0)


def clips(tmp_path: Path) -> dict[str, str]:
    """A 2 s walk (21 frames) and a 1 s jab (11 frames)."""
    return {"walk": write_clip(tmp_path / "walk.npz", 21), "jab": write_clip(tmp_path / "jab.npz", 11)}


def assert_random(times: torch.Tensor, duration: float):
    """Spread uniformly across the clip, as reset times have always been."""
    assert times.numel() > 100
    assert (times >= 0.0).all() and (times <= duration).all()
    assert times.min().item() < 0.1 * duration
    assert times.max().item() > 0.9 * duration
    assert times.mean().item() == pytest.approx(duration / 2, rel=0.1)


def test_random_start_defaults_to_on_and_shipped_manifests_still_load():
    manifests = sorted(Path(CONFIGS_DIR, "manifests").glob("*.yaml"))
    assert manifests
    for manifest in manifests:
        assert all(spec.random_start for spec in load_manifest(str(manifest))), manifest


@pytest.mark.parametrize("value", ["no", 0, 1, None, [False]])
def test_random_start_must_be_a_bool(tmp_path, value):
    manifest = write_manifest(tmp_path, {"walk": {"file": "walk.npz", "random_start": value}})
    with pytest.raises(ValueError, match="walk.random_start"):
        load_manifest(manifest)


@pytest.mark.parametrize("declared", [{}, {"random_start": True}])
def test_omitted_or_true_starts_at_a_random_time(tmp_path, declared):
    motions = MotionSet.from_manifest(write_manifest(tmp_path, {"walk": {"file": clips(tmp_path)["walk"], **declared}}), "cpu")
    assert motions.random_start.tolist() == [True]

    reset = motions.sample_reset(NUM_RESETS)

    assert_random(reset.times, 2.0)
    assert torch.equal(reset.locomotion_start_times, reset.times)
    assert torch.equal(reset.base_motion_ids, reset.motion_ids)


def test_false_starts_at_zero(tmp_path):
    files = clips(tmp_path)
    manifest = write_manifest(
        tmp_path,
        {
            "walk": {"file": files["walk"], "sample_prob": 0.5, "random_start": False},
            "run": {"file": files["walk"], "sample_prob": 0.5},
        },
    )
    motions = MotionSet.from_manifest(manifest, "cpu")
    assert motions.random_start.tolist() == [False, True]

    reset = motions.sample_reset(NUM_RESETS)

    walk = reset.motion_ids == motions.name_to_id["walk"]
    assert walk.any() and (~walk).any()
    assert (reset.times[walk] == 0.0).all()
    assert_random(reset.times[~walk], 2.0)
    assert torch.equal(reset.locomotion_start_times, reset.times)


def behavior_resets(tmp_path: Path, behavior_random_start: bool, locomotion_random_start: dict[str, bool]):
    """Resets into a 1 s jab behavior, whose locomotion return motions are a pair of 2 s walks."""
    files = clips(tmp_path)
    manifest = write_manifest(
        tmp_path,
        {
            "walk_a": {"file": files["walk"], "sample_prob": 0.1, "random_start": locomotion_random_start["walk_a"]},
            "walk_b": {"file": files["walk"], "sample_prob": 0.1, "random_start": locomotion_random_start["walk_b"]},
            "jab": {"file": files["jab"], "sample_prob": 0.8, "trigger": "binary_key_0", "random_start": behavior_random_start},
        },
    )
    motions = MotionSet.from_manifest(manifest, "cpu")
    reset = motions.sample_reset(NUM_RESETS)
    in_jab = reset.motion_ids == motions.name_to_id["jab"]
    assert in_jab.sum() > 1000
    return motions, reset, in_jab


def test_behavior_reset_return_motion_follows_its_own_setting(tmp_path):
    """A random-start jab returns to walk_a, which starts at 0, or walk_b, which starts at random:
    each return clip clock follows the locomotion motion's random_start, not the jab's."""
    motions, reset, in_jab = behavior_resets(tmp_path, True, {"walk_a": False, "walk_b": True})
    walk_a, walk_b = motions.name_to_id["walk_a"], motions.name_to_id["walk_b"]

    assert_random(reset.times[in_jab], 1.0)
    returns_to = reset.base_motion_ids[in_jab]
    assert set(returns_to.tolist()) == {walk_a, walk_b}
    start = reset.locomotion_start_times[in_jab]
    assert (start[returns_to == walk_a] == 0.0).all()
    assert_random(start[returns_to == walk_b], 2.0)


def test_behavior_reset_at_zero_still_returns_to_a_random_start(tmp_path):
    """A jab with random_start: false places the robot at its first frame, while the walk it returns
    to keeps its own default random start."""
    _, reset, in_jab = behavior_resets(tmp_path, False, {"walk_a": True, "walk_b": True})

    assert (reset.times[in_jab] == 0.0).all()
    assert_random(reset.locomotion_start_times[in_jab], 2.0)


def test_discriminator_times_stay_random(tmp_path):
    """random_start covers initialization only: discriminator batches draw times across the clip."""
    motions = MotionSet.from_manifest(
        write_manifest(tmp_path, {"walk": {"file": clips(tmp_path)["walk"], "random_start": False}}), "cpu"
    )
    assert_random(motions.library.sample_times(motions.sample_ids(NUM_RESETS)), 2.0)
