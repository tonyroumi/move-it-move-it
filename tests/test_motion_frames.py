"""Characterization of clip loading, time sampling and frame interpolation."""

import numpy as np
import pytest
import torch

from conftest import BODY_NAMES, DOF_NAMES, FPS, clip_arrays
from moveitmoveit.motion.library import MotionFrames, MotionLibrary

DT = 1.0 / FPS


def sample_at(library: MotionLibrary, clip: int, frames: float):
    """Sample one clip at a time given in frames."""
    ids = torch.tensor([clip])
    return library.frames(ids, torch.tensor([frames * DT], dtype=torch.float32))


def test_clips_are_padded_to_the_longest(clip_files):
    library = MotionLibrary(clip_files, "cpu")

    assert library.num_motions == 3
    assert library.dof_positions.shape == (3, 31, 3)
    assert library.body_positions.shape == (3, 31, 3, 3)
    assert library.num_frames.tolist() == [11, 31, 21]
    assert library.duration.tolist() == pytest.approx([10 * DT, 30 * DT, 20 * DT])
    assert library.dt == pytest.approx(DT)
    assert library.dof_names == DOF_NAMES
    assert library.body_names == BODY_NAMES
    assert library.dof_positions.dtype == torch.float32


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"fps": np.array(30.0)}, "frame rate"),
        ({"dof_names": np.array(["x", "y", "z"])}, "DOF names"),
        ({"body_names": np.array(["x", "y", "z"])}, "body names"),
    ],
)
def test_clips_must_share_skeleton_and_frame_rate(write_clip, overrides, message):
    files = [write_clip("a", 5), write_clip("b", 5, **overrides)]
    with pytest.raises(AssertionError, match=message):
        MotionLibrary(files, "cpu")


def test_missing_file_and_empty_list_are_rejected():
    with pytest.raises(AssertionError):
        MotionLibrary(["/does/not/exist.npz"], "cpu")
    with pytest.raises(AssertionError):
        MotionLibrary([], "cpu")


def test_field_order_and_shapes(clip_files):
    library = MotionLibrary(clip_files, "cpu")
    ids = torch.tensor([0, 1, 2, 1])
    out = library.frames(ids, torch.zeros(4))

    assert isinstance(out, MotionFrames)
    assert out._fields == (
        "dof_positions",
        "dof_velocities",
        "body_positions",
        "body_rotations",
        "body_linear_velocities",
        "body_angular_velocities",
    )
    assert [tensor.shape for tensor in out] == [
        (4, 3),
        (4, 3),
        (4, 3, 3),
        (4, 3, 4),
        (4, 3, 3),
        (4, 3, 3),
    ]


@pytest.mark.parametrize(
    "frames, expected_ramp",
    [
        (3.0, 3.0),  # on a frame
        (3.4, 3.4),  # rounds down: blends forward from frame 3
        (3.6, 1.4 * 4.0 - 0.4 * 5.0),  # rounds up: extrapolates backward from frames 4 and 5
        (-1.0, -1.0),  # before the clip: extrapolates from frames 0 and 1
    ],
)
def test_frame_blend_is_round_based(clip_files, frames, expected_ramp):
    library = MotionLibrary(clip_files, "cpu")
    dof_positions = sample_at(library, 0, frames).dof_positions

    assert dof_positions[0].tolist() == pytest.approx([expected_ramp * k for k in (1, 2, 3)], abs=1e-3)


def test_blend_extrapolation_is_exact_on_nonlinear_frames(write_clip):
    # On a nonlinear clip, "1.4*f[k+1] - 0.4*f[k+2]" and "0.4*f[k] + 0.6*f[k+1]" differ.
    arrays = clip_arrays(8)
    arrays["dof_positions"] = arrays["dof_positions"] ** 2
    library = MotionLibrary([write_clip("nonlinear", 8, dof_positions=arrays["dof_positions"])], "cpu")

    frames = arrays["dof_positions"]
    expected = 1.4 * frames[4] - 0.4 * frames[5]
    assert sample_at(library, 0, 3.6).dof_positions[0].tolist() == pytest.approx(expected.tolist(), abs=2e-2)


def test_times_past_the_end_hold_the_clips_own_final_frame(clip_files):
    library = MotionLibrary(clip_files, "cpu")

    # clip 0 is shorter than the padded tensor; the padded zeros must never be reached
    held = sample_at(library, 0, 400.0).dof_positions
    assert held[0].tolist() == pytest.approx([10.0, 20.0, 30.0], abs=1e-2)


def test_rotations_slerp_between_frames(clip_files):
    library = MotionLibrary(clip_files, "cpu")
    rotations = sample_at(library, 1, 2.25).body_rotations

    half_yaw = 0.01 * 2.0 * 2.25
    assert rotations[0, 0].tolist() == pytest.approx([0.0, 0.0, np.sin(half_yaw), np.cos(half_yaw)], abs=1e-4)
    assert torch.linalg.norm(rotations, dim=-1).tolist()[0] == pytest.approx([1.0, 1.0, 1.0], abs=1e-4)


def test_sample_times_are_uniform_in_each_clips_duration(clip_files):
    library = MotionLibrary(clip_files, "cpu")
    ids = torch.tensor([0, 1, 2]).repeat_interleave(5_000)

    torch.manual_seed(1)
    times = library.sample_times(ids)
    torch.manual_seed(1)
    assert torch.equal(times, library.duration[ids] * torch.rand(ids.shape[0]))

    for clip in range(3):
        assert times[ids == clip].max() <= library.duration[clip]
        assert times[ids == clip].mean().item() == pytest.approx(library.duration[clip].item() / 2, rel=0.05)


def test_history_steps_back_by_clip_dt_newest_first(clip_files):
    library = MotionLibrary(clip_files, "cpu")
    ids = torch.tensor([0, 1])
    times = torch.tensor([5.0 * DT, 9.0 * DT], dtype=torch.float32)

    history = library.history(ids, times, num_steps=3)
    assert history[0].shape == (6, 3)
    assert history[2].shape == (6, 3, 3)

    for field, data in enumerate(history):
        for step in range(3):
            expected = library.frames(ids, times - step * library.dt)[field]
            assert torch.equal(data[step::3], expected)
