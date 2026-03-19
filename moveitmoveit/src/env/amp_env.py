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

        self.motion_lib = MotionLibrary(srcs=motion_clips, skeleton=self.skeleton)

        disc_obs = self._get_disc_obs()
        self.disc_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=disc_obs.shape, dtype=np.float32,
        )

        self._current_clip_id: int = 0
        self._current_frame_id: int = 0
        self._prev_disc_obs: np.ndarray = disc_obs

        self._timestep_buf = 0
        self._time_buf = 0
        self._done_buf = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        clip_id, frame_id, frame = self.motion_lib.sample_start_state()
        self._current_clip_id = clip_id
        self._current_frame_id = frame_id
        self._episode_step = 0

        self.sim.init_from_reference_motion(
            frame["qpos"], frame.get("qvel"),
        )

        obs = self._get_obs()
        disc_obs = self._get_disc_obs() 
        # there are discriminator observations from the motion lib, but there are also 
        # discriminator observations from the environment. Discriminator discriminators between the two 
        self._prev_disc_obs = disc_obs
        info = {
            "disc_obs": disc_obs,
            "motion_frame": self._current_frame_id,
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:

        action = np.asarray(action, dtype=np.float64)

        prev_disc_obs = self._prev_disc_obs
        obs, reward, terminated, truncated, _ = super().step(action)

        self._current_frame_id += 1
        curr_disc_obs = self._get_disc_obs()
        self._prev_disc_obs = curr_disc_obs

        # here we have a bunch of stuff to do 
        # we need to check if environment has timeout based on the ep length.
        # we need to check if we have fallen to the ground
        # only fail after the first timestep
        # 

        info = {
            "prev_disc_obs": prev_disc_obs,
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

        joint_rot = self.skeleton.dof_to_rot(dof_pos)

        root_rot_norm = transforms.quat_to_tan_norm(root_rot)
        joint_rot_obs = transforms.quat_to_tan_norm(joint_rot)
        ee_rel_pos = self.sim.ee_positions - root_pos[None]
        root_height = root_pos[-1]

        obs = np.concatenate([
            root_rot_norm.ravel(),
            root_vel.ravel(),
            root_ang_vel.ravel(),
            joint_rot_obs.ravel(),
            dof_vel.ravel(),
            ee_rel_pos.ravel(),
            np.atleast_1d(root_height),
        ])
        return obs.astype(np.float32)

    def _get_disc_obs(self) -> np.ndarray:
        """Discriminator observation vector."""

        # Randomly sample motions and times from our library
        motion_ids = self.motion_lib.sample_motions(1)
        motion_times = self.motion_lib.sample_times(motion_ids)

        # Tile the number of clips / ids for the number of observation steps desired / history discriminator will see 
        motion_ids = np.tile(motion_ids.unsqueeze(-1), [1, self.params.num_disc_obs_steps])

        # Provided the timestep, obtain each state and next state for the observation window
        curr_time = -self.sim.timestep * np.arange(0, self.params.num_disc_obs_steps)
        curr_time = np.flip(curr_time, axis=[0]) # oldest -> newest (sequential order)

        motion_times = motion_times + curr_time

        (root_pos, root_rot, root_vel, root_ang_vel,
        joint_rot, joint_pos, dof_vel) = self.motion_lib.calculate_motion_frame(motion_ids, motion_times)

        ref_root_pos = root_pos[:, -1, :]
        ref_root_rot = root_rot[:, -1, :]

        root_pos_obs = root_pos - ref_root_pos

        key_pos = joint_pos[..., self.skeleton.ee_ids, :]

        key_pos -= root_pos
        root_pos_obs[..., 2] = root_pos[..., 2]

        root_rot_flat = np.reshape(root_rot, [root_rot.shape[0] * root_rot.shape[1], root_rot.shape[2]])
        root_rot_obs_flat = transforms.quat_to_tan_norm(root_rot_flat)
        root_rot_obs = np.reshape(root_rot_obs_flat, [root_rot.shape[0], root_rot.shape[1], root_rot_obs_flat.shape[-1]])

        joint_rot_flat = np.reshape(joint_rot, [joint_rot.shape[0] * joint_rot.shape[1] * joint_rot.shape[2], joint_rot.shape[3]])
        joint_rot_obs_flat = transforms.quat_to_tan_norm(joint_rot_flat)
        joint_rot_obs = np.reshape(joint_rot_obs_flat, [joint_rot.shape[0], joint_rot.shape[1], joint_rot.shape[2] * joint_rot_obs_flat.shape[-1]])

        key_pos = np.reshape(key_pos, [key_pos.shape[0], key_pos.shape[1], key_pos.shape[2] * key_pos.shape[3]])
        obs = [root_pos_obs, root_rot_obs, joint_rot_obs, key_pos, root_vel, root_ang_vel, dof_vel]

        obs = np.concatenate(obs, dim=-1)
        obs = obs.reshape(0, -1)

        return obs

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


def compute_disc_obs()