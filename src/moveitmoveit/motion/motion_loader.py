# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os

import numpy as np
import torch


class MotionLoader:
    """
    Helper class to load and sample motion data from one or more NumPy-file format motion clips.

    All motion data is stored in single tensors shared across clips, with an added leading "clip"
    dimension, e.g. ``dof_positions`` has shape ``(num_clips, max_num_frames, num_dofs)``. Clips
    shorter than the longest clip are zero-padded along the frame dimension; the padded region is
    never sampled, since frame indexing for a given clip is always clamped to that clip's own
    ``num_frames``.

    All clips must share the same DOF names, body names, and frame rate (``dt``).
    """

    def __init__(self, motion_files: list[str], device: torch.device) -> None:
        """Load one or more motion files and initialize the internal variables.

        Args:
            motion_files: Motion file paths to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If ``motion_files`` is empty, if any specified motion file doesn't
                exist, or if the loaded clips don't share the same DOF names, body names, or
                frame rate.
        """
        assert len(motion_files) > 0, "MotionLoader requires at least one motion file."
        for motion_file in motion_files:
            assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"

        self.device = device
        clips = [np.load(motion_file) for motion_file in motion_files]

        self._dof_names = clips[0]["dof_names"].tolist()
        self._body_names = clips[0]["body_names"].tolist()
        self.dt = 1.0 / clips[0]["fps"]
        for clip in clips[1:]:
            assert clip["dof_names"].tolist() == self._dof_names, "All motion clips must share the same DOF names."
            assert clip["body_names"].tolist() == self._body_names, "All motion clips must share the same body names."
            assert 1.0 / clip["fps"] == self.dt, "All motion clips must share the same frame rate (dt)."

        # number of frames per clip (N,), and the padded (max) frame count across clips
        self.num_frames = np.array([clip["dof_positions"].shape[0] for clip in clips])
        max_frames = int(self.num_frames.max())

        def stack(key: str) -> torch.Tensor:
            # pad each clip to max_frames along the frame dimension, then stack over clips
            sample_shape = clips[0][key].shape[1:]
            padded = np.zeros((len(clips), max_frames, *sample_shape), dtype=np.float32)
            for i, clip in enumerate(clips):
                padded[i, : self.num_frames[i]] = clip[key]
            return torch.tensor(padded, dtype=torch.float32, device=self.device)

        self.dof_positions = stack("dof_positions")
        self.dof_velocities = stack("dof_velocities")
        self.body_positions = stack("body_positions")
        self.body_rotations = stack("body_rotations")
        self.body_linear_velocities = stack("body_linear_velocities")
        self.body_angular_velocities = stack("body_angular_velocities")

        self.duration = self.dt * (self.num_frames - 1)
        for motion_file, num_frames, duration in zip(motion_files, self.num_frames, self.duration):
            print(f"Motion loaded ({motion_file}): duration: {duration} sec, frames: {num_frames}")

    @property
    def num_clips(self) -> int:
        """Number of motion clips."""
        return self.dof_positions.shape[0]

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        data: torch.Tensor,
        *,
        clip_indexes: torch.Tensor,
        index_0: torch.Tensor,
        index_1: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive frames, gathered from the clip tensor.

        Args:
            data: Motion data tensor. Shape is (num_clips, max_frames, X) or (num_clips, max_frames, M, X).
            clip_indexes: Clip index per sample. Shape is (N,).
            index_0: First frame index per sample (within its clip). Shape is (N,).
            index_1: Second frame index per sample (within its clip). Shape is (N,).
            blend: Interpolation coefficient between 0 (index_0) and 1 (index_1). Shape is (N,).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        a = data[clip_indexes, index_0]
        b = data[clip_indexes, index_1]
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        data: torch.Tensor,
        *,
        clip_indexes: torch.Tensor,
        index_0: torch.Tensor,
        index_1: torch.Tensor,
        blend: torch.Tensor,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation), gathered
        from the clip tensor.

        Args:
            data: Motion data tensor (wxyz quaternions). Shape is (num_clips, max_frames, 4) or
                (num_clips, max_frames, M, 4).
            clip_indexes: Clip index per sample. Shape is (N,).
            index_0: First frame index per sample (within its clip). Shape is (N,).
            index_1: Second frame index per sample (within its clip). Shape is (N,).
            blend: Interpolation coefficient between 0 (index_0) and 1 (index_1). Shape is (N,).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        q0 = data[clip_indexes, index_0]
        q1 = data[clip_indexes, index_1]
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wyzx
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(
        self, times: np.ndarray, clip_indexes: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and each sample's clip duration, to sample motion values.
                Specified times will be clipped to fall within the range of that clip's duration.
            clip_indexes: Which clip each sample is drawn from. Shape is (N,).

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        duration = self.duration[clip_indexes]
        num_frames = self.num_frames[clip_indexes]
        phase = np.clip(times / duration, 0.0, 1.0)
        index_0 = (phase * (num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_clip_indexes(self, num_samples: int) -> np.ndarray:
        """Randomly sample, per sample, which clip to draw from.

        Args:
            num_samples: Number of clip-index samples to generate.

        Returns:
            Clip indexes, uniformly sampled over ``[0, num_clips)``.
        """
        return np.random.randint(0, self.num_clips, size=num_samples)

    def sample_times(
        self, num_samples: int, clip_indexes: np.ndarray | None = None, duration: float | None = None
    ) -> np.ndarray:
        """Sample random motion times uniformly within each sample's clip duration.

        Args:
            num_samples: Number of time samples to generate.
            clip_indexes: Which clip each sample is drawn from. If not defined, clips are
                sampled uniformly at random (see :meth:`sample_clip_indexes`).
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of each sample's clip duration.

        Raises:
            AssertionError: If the specified duration is longer than a sampled clip's duration.

        Returns:
            Time samples, between 0 and the specified/clip duration.
        """
        if clip_indexes is None:
            clip_indexes = self.sample_clip_indexes(num_samples)
        clip_durations = self.duration[clip_indexes]
        if duration is not None:
            assert np.all(duration <= clip_durations), (
                f"The specified duration ({duration}) is longer than a sampled clip's duration"
            )
            clip_durations = np.full(num_samples, duration, dtype=np.float64)
        return clip_durations * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self,
        num_samples: int,
        clip_indexes: np.ndarray | None = None,
        times: np.ndarray | None = None,
        duration: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data, drawing each sample from its assigned clip.

        Args:
            num_samples: Number of samples to generate. If ``times`` is defined, this parameter
                is ignored (``len(times)`` is used instead).
            clip_indexes: Which clip each sample is drawn from. If not defined, clips are
                sampled uniformly at random (see :meth:`sample_clip_indexes`).
            times: Motion time used for sampling, per sample.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample. If not defined, samples will be
                within the range of each sample's clip duration. If ``times`` is defined,
                this parameter is ignored.

        Returns:
            A tuple containing sampled motion data:
                - DOF positions (with shape (N, num_dofs))
                - DOF velocities (with shape (N, num_dofs))
                - Body positions (with shape (N, num_bodies, 3))
                - Body rotations (with shape (N, num_bodies, 4), as wxyz quaternion)
                - Body linear velocities (with shape (N, num_bodies, 3))
                - Body angular velocities (with shape (N, num_bodies, 3))
        """
        if clip_indexes is None:
            clip_indexes = self.sample_clip_indexes(num_samples)
        if times is None:
            times = self.sample_times(num_samples, clip_indexes=clip_indexes, duration=duration)
        index_0, index_1, blend = self._compute_frame_blend(times, clip_indexes)

        clip_indexes = torch.as_tensor(clip_indexes, dtype=torch.long, device=self.device)
        index_0 = torch.as_tensor(index_0, dtype=torch.long, device=self.device)
        index_1 = torch.as_tensor(index_1, dtype=torch.long, device=self.device)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, clip_indexes=clip_indexes, index_0=index_0, index_1=index_1, blend=blend),
            self._interpolate(self.dof_velocities, clip_indexes=clip_indexes, index_0=index_0, index_1=index_1, blend=blend),
            self._interpolate(self.body_positions, clip_indexes=clip_indexes, index_0=index_0, index_1=index_1, blend=blend),
            self._slerp(self.body_rotations, clip_indexes=clip_indexes, index_0=index_0, index_1=index_1, blend=blend),
            self._interpolate(
                self.body_linear_velocities, clip_indexes=clip_indexes, index_0=index_0, index_1=index_1, blend=blend
            ),
            self._interpolate(
                self.body_angular_velocities, clip_indexes=clip_indexes, index_0=index_0, index_1=index_1, blend=blend
            ),
        )

    def sample_history(
        self,
        num_samples: int,
        clip_indexes: np.ndarray | None = None,
        times: np.ndarray | None = None,
        num_steps: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a history of motion states, ``num_steps`` apart in time, ending at each sample's time.

        Args:
            num_samples: Number of samples to generate.
            clip_indexes: Which clip each sample is drawn from. If not defined, clips are
                sampled uniformly at random (see :meth:`sample_clip_indexes`).
            times: Motion time used for sampling, per sample (the most recent time in each
                sample's history). If not defined, motion data will be random sampled uniformly
                in time.
            num_steps: Number of historical steps to sample per sample, spaced ``dt`` apart,
                going backwards in time from ``times``.

        Returns:
            A tuple containing sampled motion data (see :meth:`sample`), flattened over samples
            and steps, i.e. with shape (num_samples * num_steps, ...).
        """
        if clip_indexes is None:
            clip_indexes = self.sample_clip_indexes(num_samples)
        if times is None:
            times = self.sample_times(num_samples, clip_indexes)
        # step back num_steps times, dt apart, from each sample's time -> (num_samples * num_steps,)
        history_times = (np.expand_dims(times, axis=-1) - self.dt * np.arange(0, num_steps)).flatten()
        history_clip_indexes = np.repeat(clip_indexes, num_steps)
        return self.sample(num_samples=num_samples * num_steps, clip_indexes=history_clip_indexes, times=history_times)

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader([args.file], "cpu")

    print("- number of frames:", motion.num_frames[0])
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)
