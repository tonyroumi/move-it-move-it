"""
Reference-motion storage and sampling.

Loads pre-processed motion clips and provides random-access frame
retrieval for both the environment (sequential playback) and the
discriminator (uniform random sampling).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from moveitmoveit.src.sim import Skeleton
import moveitmoveit.src.transforms as transforms

@dataclass
class MotionClip:
    """
    A single reference-motion sequence.

    All arrays are indexed by frame number along axis 0.
    """
    fps: float
    position: np.ndarray    # (T, J, 3) , cm
    rotation: np.ndarray    # (T, J, 4) , (x, y, z, w) radians

    @property
    def num_frames(self) -> int:
        return self.position.shape[0]

    @property
    def dt(self) -> float:
        return 1.0 / self.fps

    @property
    def duration(self) -> float:
        return self.num_frames * self.dt

    def get_frame(self, idx: int) -> dict:
        """Return all quantities for a single frame."""
        idx = idx % self.num_frames
        frame = {
            "positions": self.position[idx],
            "rotations": self.rotation[idx]
        }
        return frame

    def get_frame_pair(self, idx: int) -> Tuple[dict, dict]:
        """
        Return the frame at *idx* and the next frame (for finite-difference
        velocity estimation or transition pairs).
        """
        return self.get_frame(idx), self.get_frame(idx + 1)


class MotionLibrary:
    """
    Collection of motion clips with weighted random sampling.

    Parameters
    ----------
    clips : list[MotionClip]
        One or more motion clips.
    weights : list[float], optional
        Sampling weight for each clip (normalised internally).
    """

    def __init__(
        self,
        clips: List[MotionClip],
        skeleton: Skeleton,
    ) -> None:
        assert len(clips) > 0
        self._clips = clips

        # Weight proportional to duration. why? TODO
        weights = [c.duration for c in clips]
        w = np.array([weights], dtype=np.float64)
        self._weights = w / w.sum()

        self._skeleton = skeleton

        self._load()
    
    def _load(self):

        frame_root_pos = []
        frame_root_rot = []
        frame_root_vel = []
        frame_root_ang_vel = []
        frame_joint_rot = []
        frame_dof_vel = []

        for clip in self._clips:
            fps = clip.fps
            dt = 1 / fps

            root_pos = np.array(clip.position[:, 0, :])
            root_rot = np.array(clip.rotation[:, 0, :])
            rotations = np.array(clip.rotation[:, 1:, :])
            rotations = transforms.quat_pos(rotations)

            root_vel = np.zeros_like(root_pos)
            root_vel[:-1, :] = fps * (root_pos[1:, :] - root_pos[:-1, :])
            root_vel[-1, :] = root_vel[-2, :]

            root_ang_vel = np.zeros_like(root_pos)
            root_drot = transforms.quat_diff(root_rot[:-1, :], root_rot[1:, :])
            root_ang_vel[:-1, :] = clip.fps * transforms.quat_to_exp_map(root_drot)
            root_ang_vel[-1, :] = root_ang_vel[-2, :]

            dof_vel = self._skeleton.compute_dof_vel(rotations, dt, transforms)

            frame_root_pos.append(root_pos)
            frame_root_rot.append(root_rot)
            frame_root_vel.append(root_vel)
            frame_root_ang_vel.append(root_ang_vel)
            frame_joint_rot.append(rotations)
            frame_dof_vel.append(dof_vel)

        self._frame_root_pos = np.concatenate(frame_root_pos, axis=0)
        self._frame_root_rot = np.concatenate(frame_root_rot, axis=0)
        self._frame_root_vel = np.concatenate(frame_root_vel, axis=0)
        self._frame_root_ang_vel = np.concatenate(frame_root_ang_vel, axis=0)
        self._frame_joint_rot = np.concatenate(frame_joint_rot, axis=0)
        self._frame_dof_vel = np.concatenate(frame_dof_vel, axis=0)

        self._motion_ids = np.arange(self.num_clips)

    @property
    def num_clips(self) -> int:
        return len(self._clips)

    def get_clip(self, idx: int) -> MotionClip:
        return self._clips[idx]

    def sample_frames(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Uniformly sample *n* (clip_index, frame_index) pairs, weighted
        by clip duration.

        Returns
        -------
        clip_ids  : (n,) int
        frame_ids : (n,) int
        """
        if rng is None:
            rng = np.random.default_rng()

        clip_ids = rng.choice(len(self._clips), size=n, p=self._weights)
        frame_ids = np.empty(n, dtype=np.int64)
        for i, cid in enumerate(clip_ids):
            frame_ids[i] = rng.integers(0, self._clips[cid].num_frames)
        return clip_ids, frame_ids

    def sample_frame_data(
        self,
        n: int,
        rng: Optional[np.random.Generator] = None,
    ) -> List[dict]:
        """
        Sample *n* random frames and return their data dicts.
        """
        clip_ids, frame_ids = self.sample_frames(n, rng)
        return [
            self._clips[cid].get_frame(fid)
            for cid, fid in zip(clip_ids, frame_ids)
        ]

    def sample_start_state(
        self,
        rng: Optional[np.random.Generator] = None,
    ) -> Tuple[int, int, dict]:
        """
        Sample a single random starting state for episode initialisation.

        Returns
        -------
        clip_id : int
        frame_id : int
        frame : dict
        """
        cids, fids = self.sample_frames(1, rng)
        return int(cids[0]), int(fids[0]), self._clips[cids[0]].get_frame(fids[0])