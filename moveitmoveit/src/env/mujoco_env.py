from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from moveitmoveit.src.sim import Skeleton, MujocoInterface, SimParams


@dataclass(frozen=True)
class MujocoEnvParams(SimParams):
    max_episode_time: float = 10.0  # seconds


class MujocoEnv(gym.Env):
    """Base MuJoCo environment with a gym-compatible interface."""

    metadata = {"render_modes": []}

    def __init__(self, model_path: str | Path, params: MujocoEnvParams) -> None:
        super().__init__()
        self._mj_model = mujoco.MjModel.from_xml_path(str(model_path))
        self._mj_data = mujoco.MjData(self._mj_model)

        self.skeleton = Skeleton(self._mj_model)
        self.sim = MujocoInterface(self.skeleton, self._mj_data, params=params)
        self.params = params

        self._episode_step = 0

        a_low, a_high = self.skeleton.compute_action_bounds()
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32,
        )

    @property
    def episode_time(self) -> float:
        """Elapsed time in seconds for the current episode."""
        return self._episode_step / self.params.control_freq

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self._episode_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.sim.step(self._action_to_ctrl(action))
        self._episode_step += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.episode_time >= self.params.max_episode_time

        return obs, reward, terminated, truncated, {}

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map policy action to simulator control signals."""
        return action

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        """Compute and return the observation vector."""
        ...

    @abstractmethod
    def _compute_reward(self) -> float:
        """Compute and return the step reward."""
        ...

    @abstractmethod
    def _check_termination(self) -> bool:
        """Return True if the episode should terminate early."""
        ...
