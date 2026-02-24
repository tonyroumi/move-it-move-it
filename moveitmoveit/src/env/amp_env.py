from __future__ import annotations

from pathlib import Path
from typing import Tuple, List

import numpy as np
from gymnasium import spaces

from moveitmoveit.src.motion import MotionLibrary
import moveitmoveit.src.transforms as transforms

from .mujoco_env import MujocoEnv, MujocoEnvParams


class AMPEnv(MujocoEnv):
    """Combined goal + style RL environment."""

    def __init__(
        self,
        model_path: str | Path,
        motion_clips: List[str],
        params: MujocoEnvParams = MujocoEnvParams(),
    ) -> None:
        super().__init__(model_path, params)

        self.motion_lib = None  # MotionLibrary(srcs=motion_clips)

        disc_obs = self._get_disc_obs()
        self.disc_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=disc_obs.shape, dtype=np.float32,
        )

        self._current_clip_id: int = 0
        self._current_frame_id: int = 0
        self._prev_disc_obs: np.ndarray = disc_obs
        self._rng = np.random.default_rng()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        # if seed is not None:
        #     self._rng = np.random.default_rng(seed)

        # clip_id, frame_id, frame = self.motion_lib.sample_start_state(self._rng)
        # self._current_clip_id = clip_id
        # self._current_frame_id = frame_id
        # self._episode_step = 0

        # self.sim.init_from_reference_motion(
        #     frame["qpos"], frame.get("qvel"),
        # )

        obs = self._get_obs()
        disc_obs = self._get_disc_obs()
        self._prev_disc_obs = disc_obs
        info = {
            "disc_obs": disc_obs,
            "motion_frame": self._current_frame_id,
        }
        return obs, info

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        action = np.asarray(action, dtype=np.float64)

        prev_disc_obs = self._prev_disc_obs
        obs, reward, terminated, truncated, _ = super().step(action)

        self._current_frame_id += 1
        curr_disc_obs = self._get_disc_obs()
        self._prev_disc_obs = curr_disc_obs

        info = {
            "prev_disc_obs": prev_disc_obs,
            "disc_obs": curr_disc_obs,
            "motion_frame": self._current_frame_id,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        """Policy observation vector."""
        root_pos = self.sim.root_pos
        root_rot = self.sim.root_quat
        root_vel = self.sim.root_vel
        root_ang_vel = self.sim.root_ang_vel
        joint_rot = self.sim.joint_quat
        dof_vel = self.sim.dof_vel

        root_rot_norm = transforms.quat_to_tan_norm(root_rot)
        joint_rot_obs = transforms.quat_to_tan_norm(joint_rot)
        ee_rel_pos = root_pos[None] - self.sim.ee_positions
        root_height = root_pos[-1]

        obs = np.concatenate([
            np.atleast_1d(root_height),
            root_rot_norm.ravel(),
            root_vel.ravel(),
            root_ang_vel.ravel(),
            joint_rot_obs.ravel(),
            dof_vel.ravel(),
            ee_rel_pos.ravel(),
        ])
        return obs.astype(np.float32)

    def _get_disc_obs(self) -> np.ndarray:
        """Discriminator observation vector."""
        # TODO: implement discriminator features
        return self._get_obs()

    def _compute_reward(self) -> float:
        """Task / goal reward. Override in subclasses."""
        return 0.0

    def _check_termination(self) -> bool:
        """Early termination conditions (e.g. root height, deviation)."""
        return False

    def get_reference_frame(self) -> dict:
        """Return the current reference-motion frame."""
        clip = self.motion_lib.get_clip(self._current_clip_id)
        return clip.get_frame(self._current_frame_id)
