"""Shared fixtures. Run with: python -m pytest tests

Modules that need IsaacLab skip themselves without it. Under IsaacLab's Python
(./isaaclab.sh -p -m pytest tests) they run too; no simulator is started either way.

Clips are synthetic so these tests pin code semantics, not the contents of the committed clips.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]

DOF_NAMES = ["dof_a", "dof_b", "dof_c"]
BODY_NAMES = ["torso", "hand", "foot"]
FPS = 60.0


def clip_arrays(num_frames: int, scale: float = 1.0, fps: float = FPS) -> dict[str, np.ndarray]:
    """A clip whose every field ramps linearly with the frame index, so blends are easy to predict."""
    ramp = scale * np.arange(num_frames, dtype=np.float32)
    num_dofs, num_bodies = len(DOF_NAMES), len(BODY_NAMES)

    # xyzw yaw rotation, slowly increasing per frame
    half_yaw = 0.01 * ramp
    rotations = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
    rotations[..., 2] = np.sin(half_yaw)[:, None]
    rotations[..., 3] = np.cos(half_yaw)[:, None]

    dof_offsets = np.arange(1, num_dofs + 1, dtype=np.float32)
    body_offsets = np.arange(1, num_bodies * 3 + 1, dtype=np.float32).reshape(num_bodies, 3)
    return {
        "fps": np.array(fps),
        "dof_names": np.array(DOF_NAMES),
        "body_names": np.array(BODY_NAMES),
        "dof_positions": ramp[:, None] * dof_offsets,
        "dof_velocities": 2.0 * ramp[:, None] * dof_offsets,
        "body_positions": ramp[:, None, None] * body_offsets,
        "body_rotations": rotations,
        "body_linear_velocities": 3.0 * ramp[:, None, None] * body_offsets,
        "body_angular_velocities": 4.0 * ramp[:, None, None] * body_offsets,
    }


@pytest.fixture
def write_clip(tmp_path):
    def _write(name: str, num_frames: int, scale: float = 1.0, **overrides) -> str:
        path = tmp_path / f"{name}.npz"
        np.savez(path, **{**clip_arrays(num_frames, scale), **overrides})
        return str(path)

    return _write


@pytest.fixture
def clip_files(write_clip) -> list[str]:
    """Three clips of different lengths: index 0 is shorter than the padded length."""
    return [write_clip("short", 11), write_clip("long", 31, scale=2.0), write_clip("medium", 21, scale=0.5)]


@pytest.fixture
def write_manifest(tmp_path):
    def _write(motions: dict, **top_level) -> str:
        path = tmp_path / "manifest.yaml"
        with open(path, "w") as f:
            yaml.safe_dump({"motions": motions, **top_level}, f, sort_keys=False)
        return str(path)

    return _write
