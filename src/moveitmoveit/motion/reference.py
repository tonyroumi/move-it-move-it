from typing import Callable, NamedTuple

import torch

from moveitmoveit.mdp import BodyState
from moveitmoveit.utils import transforms

from .library import MotionFrames, MotionLibrary


class SkeletonMap(NamedTuple):
    """Where the robot's DOFs and bodies live in the clips' tensors."""

    dof_indexes: list[int]
    root_body_index: int
    key_body_indexes: list[int]


class ResetFrames(NamedTuple):
    """Reference Robot state to reset to."""

    dof_positions: torch.Tensor
    dof_velocities: torch.Tensor
    root_positions: torch.Tensor
    root_rotations: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor


class TrackingTargets(NamedTuple):
    dof_positions: torch.Tensor
    dof_velocities: torch.Tensor
    root_positions: torch.Tensor
    root_rotations: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor
    key_body_positions: torch.Tensor


class ReferenceMotions:
    def __init__(
        self,
        picker: Callable[[int], torch.Tensor],
        library: MotionLibrary,
        skeleton_map: SkeletonMap,
        num_amp_steps: int,
        random_yaw: bool,
        style_obs: Callable[[BodyState, bool], torch.Tensor],
    ):
        self._pick = picker
        self._library = library
        self._skeleton = skeleton_map
        self._num_amp_steps = num_amp_steps
        self._random_yaw = random_yaw
        self._style_obs = style_obs

    def frames_for_reset(self, motion_ids: torch.Tensor, times: torch.Tensor) -> ResetFrames:
        """The state to reset each env to, with no yaw applied."""
        frames = self._library.frames(motion_ids, times)
        root = self._skeleton.root_body_index
        return ResetFrames(
            frames.dof_positions[:, self._skeleton.dof_indexes],
            frames.dof_velocities[:, self._skeleton.dof_indexes],
            frames.body_positions[:, root],
            frames.body_rotations[:, root],
            frames.body_linear_velocities[:, root],
            frames.body_angular_velocities[:, root],
        )

    def tracking_targets(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
        yaw: torch.Tensor,
    ) -> TrackingTargets:
        """Reference pose and velocities at each env's motion time and yaw."""
        frames = self._library.frames(motion_ids, times)

        if self._random_yaw:
            frames = self._apply_yaw(frames, yaw)

        skeleton = self._skeleton

        return TrackingTargets(
            dof_positions=frames.dof_positions[:, skeleton.dof_indexes],
            dof_velocities=frames.dof_velocities[:, skeleton.dof_indexes],
            root_positions=frames.body_positions[:, skeleton.root_body_index],
            root_rotations=frames.body_rotations[:, skeleton.root_body_index],
            root_linear_velocities=frames.body_linear_velocities[:, skeleton.root_body_index],
            root_angular_velocities=frames.body_angular_velocities[:, skeleton.root_body_index],
            key_body_positions=frames.body_positions[:, skeleton.key_body_indexes],
        )

    def amp_seed(self, motion_ids: torch.Tensor, times: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
        observations = self._amp_observations(motion_ids, times, yaw if self._random_yaw else None)
        return observations.view(motion_ids.shape[0], self._num_amp_steps, -1)

    def style_batch(self, num_samples: int) -> torch.Tensor:
        """A batch of real motion for the discriminator"""
        motion_ids = self._pick(num_samples)
        times = self._library.sample_times(motion_ids)

        yaw = transforms.random_yaw_orientation(num_samples, self._library.device) if self._random_yaw else None
        return self._amp_observations(motion_ids, times, yaw).view(num_samples, -1)

    def _amp_observations(self, motion_ids: torch.Tensor, times: torch.Tensor, yaw: torch.Tensor | None) -> torch.Tensor:
        """Style observations of the clip history ending at `times`, optionally yawed."""
        frames = self._library.history(motion_ids, times, self._num_amp_steps)
        if yaw is not None:
            frames = self._apply_yaw(frames, yaw.repeat_interleave(self._num_amp_steps, dim=0))

        root = self._skeleton.root_body_index
        state = BodyState(
            frames.dof_positions[:, self._skeleton.dof_indexes],
            frames.dof_velocities[:, self._skeleton.dof_indexes],
            frames.body_positions[:, root],
            frames.body_rotations[:, root],
            frames.body_linear_velocities[:, root],
            frames.body_angular_velocities[:, root],
            frames.body_positions[:, self._skeleton.key_body_indexes],
        )
        return self._style_obs(state, self._random_yaw)

    def _apply_yaw(self, frames: MotionFrames, yaw: torch.Tensor) -> MotionFrames:
        body_positions, body_rotations, body_linear_velocities, body_angular_velocities = transforms.apply_random_yaw(
            yaw,
            self._skeleton.root_body_index,
            frames.body_positions,
            frames.body_rotations,
            frames.body_linear_velocities,
            frames.body_angular_velocities,
        )
        return frames._replace(
            body_positions=body_positions,
            body_rotations=body_rotations,
            body_linear_velocities=body_linear_velocities,
            body_angular_velocities=body_angular_velocities,
        )
