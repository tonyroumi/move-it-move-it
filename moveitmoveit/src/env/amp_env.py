from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
from gymnasium import spaces

from moveitmoveit.src.motion import MotionLibrary
import moveitmoveit.src.transforms as transforms
from utils import Logger

from .mujoco_env import MujocoEnv, MujocoEnvParams

@dataclass(frozen=True)
class AmpEnvParams(MujocoEnvParams):
    num_disc_obs_steps: int = 10

class AMPEnv(MujocoEnv):
    """Combined goal + style RL environment."""

    def __init__(
        self,
        model_path: str,
        motion_clips: List[str],
        params: AmpEnvParams,
        logger: Logger
    ) -> None:
        super().__init__(model_path, params)

        self.motion_lib = MotionLibrary(srcs=motion_clips, skeleton=self.skeleton, logger=logger)

        disc_obs = self._get_disc_obs()
        self.disc_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=disc_obs.shape, dtype=np.float32,
        )

        self._current_clip_id = 0
        self._current_frame_id = 0
        self._motion_offset = 0.0

        self._timestep_buf = 0
        self._time_buf = 0
        self._done = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        qpos, qvel, frame_info = self.motion_lib.sample_start_state(rng=self.np_random)
        self._current_clip_id = frame_info["clip_id"]
        self._current_frame_id = frame_info["frame_id"]
        self._motion_time = frame_info["motion_time"]

        self.sim.init_from_reference_motion(
            qpos, qvel,
        )

        self._timestep_buf = self._motion_time / self.sim.timestep #TODO(anthony) need to verify this.
        self._time_buf = self._motion_time
        self._done = 0

        obs = self._get_obs()

        # Discriminator observations are mixed with simulator state and motion lib window... TODO(better notes)
        disc_obs = self._fetch_ref_disc_obs(frame_info["clip_id"], frame_info["motion_time"]) 

        info = {
            "disc_obs": disc_obs,
            "clip_id": self._current_clip_id,
            "frame_id": self._current_frame_id,
        }
        return obs, info
    
    def _fetch_ref_disc_obs(self, clip_id: np.ndarray, motion_time: np.ndarray) -> np.ndarray:
        """WHY DO WE DO THIS CLARIFYYYYYY"""
        motion_ids = np.tile(clip_id[..., np.newaxis], [1, self.params.num_disc_obs_steps])

        # # Provided the timestep, obtain each state and next state for the observation window
        curr_time = -self.sim.timestep * np.arange(0, self.params.num_disc_obs_steps)
        curr_time = np.flip(curr_time, axis=[0]) # oldest -> newest (sequential order)

        motion_times = motion_time + curr_time

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = (
            self.motion_lib.get_frame_data(motion_ids, motion_times)
        )

        root_rot_norm = transforms.quat_to_tan_norm(root_rot)
        joint_rot_norm = transforms.quat_to_tan_norm(joint_rot)
        ee_rel_pos = body_pos[:, :, self.skeleton.ee_ids] - root_pos[:, :, np.newaxis, :]
        root_height = root_pos[:, :, -1]

        disc_obs = np.concatenate([
            root_rot_norm.ravel(),
            root_vel.ravel(),
            root_ang_vel.ravel(),
            joint_rot_norm.ravel(),
            dof_vel.ravel(),
            ee_rel_pos.ravel(),
            root_height.ravel(),
        ])
        return disc_obs.astype(np.float32)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        obs, reward, terminated, truncated, _ = super().step(action)

        self._timestep_buf += 1
        self._time_buf = self.sim.timestep * self._timestep_buf

        self._current_frame_id += 1
        curr_disc_obs = self._get_disc_obs()
        self._prev_disc_obs = curr_disc_obs

        info = {
            "disc_obs": curr_disc_obs,
            "motion_frame": self._current_frame_id,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        root_pos = self.sim.root_pos
        root_rot = self.sim.root_quat
        root_vel = self.sim.root_vel
        root_ang_vel = self.sim.root_ang_vel
        dof_pos = self.sim.dof_pos
        dof_vel = self.sim.dof_vel
        ee_pos = self.sim.ee_positions

        joint_rot = self.skeleton.dof_to_rot(dof_pos[np.newaxis])

        root_rot_norm = transforms.quat_to_tan_norm(root_rot)
        joint_rot_norm = transforms.quat_to_tan_norm(joint_rot)
        ee_rel_pos = ee_pos - root_pos[np.newaxis]
        root_height = root_pos[-1]

        obs = np.concatenate([
            root_rot_norm.ravel(),
            root_vel.ravel(),
            root_ang_vel.ravel(),
            joint_rot_norm.ravel(),
            dof_vel.ravel(),
            ee_rel_pos.ravel(),
            np.atleast_1d(root_height),
        ])
        return obs.astype(np.float32)

    def _get_disc_obs(self) -> np.ndarray:
        """Discriminator observation vector."""
        return self._get_obs()

    def _compute_reward(self) -> float:
        """Task reward: root linear velocity along facing direction (heading-frame +x)."""
        root_vel = self.sim.root_vel
        root_quat = self.sim.root_quat
        vel_heading = transforms.transform_velocities_to_root_frame(root_vel, root_quat)
        return float(vel_heading[0])

    def _check_termination(self) -> bool:
        """Early termination conditions (e.g. root height, deviation)."""
        if (self.sim.root_pos[-1] < 0.8):
            return True
        return False