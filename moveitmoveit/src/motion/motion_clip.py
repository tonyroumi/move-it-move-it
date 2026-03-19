from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

@dataclass
class MotionClip:
    """ A single motion sequence for learning. """

    name: str
    fps: float
    positions: np.ndarray  # (T, J, 3) cm
    rotations: np.ndarray  # (T, J, 4) (x, y, z, w)

    def __post_init__(self) -> None:
        if self.positions.ndim != 3 or self.positions.shape[2] != 3:
            raise ValueError(
                f"positions must be (T, J, 3), got {self.positions.shape}"
            )
        if self.rotations.ndim != 3 or self.rotations.shape[2] != 4:
            raise ValueError(
                f"rotations must be (T, J, 4), got {self.rotations.shape}"
            )
        if self.positions.shape[:2] != self.rotations.shape[:2]:
            raise ValueError(
                f"positions and rotations must share (T, J) dims: "
                f"{self.positions.shape[:2]} vs {self.rotations.shape[:2]}"
            )

    def get_frame(self, idx: int) -> dict:
        idx = idx % self.num_frames
        frame = {
            "positions": self.positions[idx],
            "rotations": self.rotations[idx]
        }
        return frame

    def get_frame_pair(self, idx: int) -> Tuple[dict, dict]:
        return self.get_frame(idx), self.get_frame(idx + 1)

    @property
    def num_frames(self) -> int:
        return int(self.positions.shape[0])

    @property
    def num_joints(self) -> int:
        return int(self.positions.shape[1])

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def duration(self) -> float:
        return self.num_frames * self.dt

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            str(path),
            name=np.array(self.name),
            fps=np.array(self.fps, dtype=np.float32),
            positions=self.positions.astype(np.float32),
            rotations=self.rotations.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str | Path) -> "MotionClip":
        d = np.load(str(path), allow_pickle=False)
        # TODO In our motion repr class add a loop mode. from MimicKit they have WRAP and CLAMP. 
        # for now we assume all of our motion clips are clamped. (they end)
        return cls(
            name=str(d["name"]),
            fps=float(d["fps"]),
            positions=d["positions"].astype(np.float32),
            rotations=d["rotations"].astype(np.float32),
        )