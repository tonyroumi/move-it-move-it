"""A manifest motion's `loop` wraps its tracking clip; behaviors still complete at phase 1. No simulator."""

from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from moveitmoveit.behavior import LOCOMOTION, BehaviorManager, reference_times, settled
from moveitmoveit.manifest import load_manifest
from moveitmoveit.motion.library import MotionLibrary
from moveitmoveit.motion_set import MotionSet
from moveitmoveit.utils.paths import CONFIGS_DIR

FPS = 10
NUM_DOFS = 2


def write_clip(path: Path, num_frames: int) -> str:
    """A clip whose DOF positions read their own frame index, so a pose names the frame it came from."""
    frames = np.arange(num_frames, dtype=np.float32)
    rotations = np.zeros((num_frames, 1, 4), dtype=np.float32)
    rotations[..., 3] = 1.0
    np.savez(
        path,
        fps=np.array(FPS),
        dof_names=np.array(["a", "b"]),
        body_names=np.array(["torso"]),
        dof_positions=np.repeat(frames[:, None], NUM_DOFS, axis=1),
        dof_velocities=np.zeros((num_frames, NUM_DOFS), dtype=np.float32),
        body_positions=np.zeros((num_frames, 1, 3), dtype=np.float32),
        body_rotations=rotations,
        body_linear_velocities=np.zeros((num_frames, 1, 3), dtype=np.float32),
        body_angular_velocities=np.zeros((num_frames, 1, 3), dtype=np.float32),
    )
    return str(path)


def write_manifest(tmp_path: Path, motions: dict) -> str:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"motions": motions}, sort_keys=False))
    return str(path)


@pytest.fixture
def motions(tmp_path) -> MotionSet:
    """A 1 s walk (11 frames) and a 1 s triggered behavior, both declaring loop: true, and a 1 s
    walk that does not."""
    walk = write_clip(tmp_path / "walk.npz", 11)
    jab = write_clip(tmp_path / "jab.npz", 11)
    manifest = write_manifest(
        tmp_path,
        {
            "walk": {"file": walk, "sample_prob": 0.4, "loop": True},
            "walk_once": {"file": walk, "sample_prob": 0.4},
            "jab": {"file": jab, "sample_prob": 0.2, "trigger": "binary_key_0", "loop": True},
        },
    )
    return MotionSet.from_manifest(manifest, "cpu")


def tracked_frame(motions: MotionSet, motion_ids, times) -> torch.Tensor:
    """The frame index of each tracked pose."""
    return motions.library.frames(torch.as_tensor(motion_ids), torch.as_tensor(times)).dof_positions[:, 0]


def test_loop_defaults_to_off_and_shipped_manifests_still_load():
    manifests = sorted(Path(CONFIGS_DIR, "manifests").glob("*.yaml"))
    assert manifests
    for manifest in manifests:
        assert not any(spec.loop for spec in load_manifest(str(manifest))), manifest


def test_loop_must_be_a_bool(tmp_path):
    manifest = write_manifest(tmp_path, {"walk": {"file": "walk.npz", "loop": "yes"}})
    with pytest.raises(ValueError, match="walk.loop"):
        load_manifest(manifest)


def test_loop_table(motions):
    assert motions.loop.tolist() == [True, False, True]


def test_loop_wraps_and_no_loop_clamps(motions):
    """Two locomotion envs 1.3 s into a 1 s clip: the looping one tracks frame 3 of its second
    pass, the other holds the last frame, as before."""
    walk, walk_once = motions.name_to_id["walk"], motions.name_to_id["walk_once"]
    motion_ids = torch.tensor([walk, walk_once])
    behaviors = BehaviorManager(2, motions.num_behaviors, "cpu")
    time = torch.full((2,), 1.3)

    times = reference_times(
        time, behaviors, torch.zeros(2), motions.library.duration[motion_ids], motions.loop[motion_ids]
    )

    assert times.tolist() == pytest.approx([0.3, 1.3])
    assert tracked_frame(motions, motion_ids, times).tolist() == pytest.approx([3.0, 10.0])


def test_loop_repeats_every_clip_duration(motions):
    walk = motions.name_to_id["walk"]
    behaviors = BehaviorManager(1, motions.num_behaviors, "cpu")
    for elapsed, frame in [(0.5, 5.0), (1.5, 5.0), (2.5, 5.0), (9.7, 7.0)]:
        times = reference_times(
            torch.tensor([elapsed]),
            behaviors,
            torch.tensor([0.0]),
            motions.library.duration[[walk]],
            motions.loop[[walk]],
        )
        assert tracked_frame(motions, [walk], times).item() == pytest.approx(frame), elapsed


def run_behavior(motions: MotionSet, dof_speed: float) -> tuple[list[float], float]:
    """Step one env the way MotionLearningEnv._update_behaviors does: press the jab at 0.2 s, then
    complete it at phase 1 or when it settles. Returns the clip times it tracked while running, and
    the time it returned to locomotion."""
    dt = 0.1
    behaviors = BehaviorManager(1, motions.num_behaviors, "cpu")
    walk = motions.name_to_id["walk"]
    base_motion_ids = torch.tensor([walk])
    motion_ids = base_motion_ids.clone()
    dof_velocities = torch.full((1, NUM_DOFS), dof_speed)

    tracked = []
    for step in range(40):
        time = torch.tensor([step * dt])
        if behaviors.active.any():
            tracked.append(
                reference_times(
                    time,
                    behaviors,
                    torch.zeros(1),
                    motions.library.duration[motion_ids],
                    motions.loop[motion_ids],
                ).item()
            )

        keys = torch.tensor([[1.0 if step == 2 else 0.0]])
        phase = behaviors.phase(time, motions.library.duration[motion_ids])
        stable = settled(
            phase,
            dof_velocities,
            motions.settle_min_phase[motion_ids],
            motions.settle_max_dof_speed[motion_ids],
        )
        was_active = bool(behaviors.active.item())
        behaviors.update(keys, (phase >= 1.0) | stable, time)
        mode = behaviors.mode
        motion_ids = torch.where(mode != LOCOMOTION, motions.behavior_motion_ids[mode.clamp_min(0)], base_motion_ids)
        if was_active and not behaviors.active.item():
            return tracked, time.item()
    raise AssertionError("the behavior never completed")


def test_looping_behavior_still_completes_at_phase_one(motions):
    """The jab declares loop: true, yet it runs once on the unwrapped clock and returns to
    locomotion at phase 1 (0.2 s + its 1 s). Its target reaches the clip's end, 1 s, the step
    before, rather than wrapping back to the first frame."""
    tracked, completed_at = run_behavior(motions, dof_speed=0.0)

    assert completed_at == pytest.approx(1.2)
    assert tracked == pytest.approx([0.1 * (i + 1) for i in range(10)])


def test_looping_behavior_still_settles(tmp_path):
    walk = write_clip(tmp_path / "walk.npz", 11)
    jab = write_clip(tmp_path / "jab.npz", 11)
    manifest = write_manifest(
        tmp_path,
        {
            "walk": {"file": walk, "sample_prob": 0.8, "loop": True},
            "jab": {
                "file": jab,
                "sample_prob": 0.2,
                "trigger": "binary_key_0",
                "loop": True,
                "settle": {"min_phase": 0.5, "max_dof_speed": 0.1},
            },
        },
    )
    motions = MotionSet.from_manifest(manifest, "cpu")

    _, completed_at = run_behavior(motions, dof_speed=0.0)
    assert completed_at == pytest.approx(0.7)

    _, completed_at = run_behavior(motions, dof_speed=1.0)
    assert completed_at == pytest.approx(1.2)
