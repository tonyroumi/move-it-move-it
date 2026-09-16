from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import collections

import torch
import torch.optim
import numpy as np

from isaaclab.envs import DirectRLEnv

from moveitmoveit.resources import RunningStandardScaler
from moveitmoveit.utils.logger import Logger


@dataclass(kw_only=True)
class BaseCfg(ABC):
    class_type: str


class BaseAgent(ABC):
    """ Abstract base class for a learning agent. """

    def __init__(
        self,
        cfg: BaseCfg,
        logger: Logger,
    ):
        self.cfg = cfg
        self.logger = logger

        self._update_step = 0

        # Episode reward/length tracking.
        self._cumulative_rewards: torch.Tensor | None = None
        self._cumulative_timesteps: torch.Tensor | None = None
        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self.mean_episode_reward: float | None = None

        # Reward term tracking
        self._reward_term_sum: dict[str, float] = collections.defaultdict(float)
        self._reward_term_count: dict[str, int] = collections.defaultdict(int)

    def init(self, env: DirectRLEnv, cfg: dict) -> None:
        """Initialize the agent with the environment and configuration."""
        self._step_dt = env.unwrapped.step_dt # policy frequency
        self._motion_names = env.unwrapped._motion_manager.motion_names
        self._cmd_dim = env.unwrapped.command_dim

        self._obs_preprocessor = self._build_observations_preprocessor(env)
        self._action_preprocessor = self._build_action_preprocessor(env)

        self._initialize_models(env, cfg["models"])
        self._initialize_optimizer()
        self._initialize_storage(env)

    def _build_observations_preprocessor(self, env: DirectRLEnv) -> RunningStandardScaler:
        """Normalizes the environment's raw observation via running statistics."""
        obs_size = env.observation_space.shape[-1]
        return RunningStandardScaler(
            size=obs_size, unscaled_dims=self._cmd_dim
        ).to(env.unwrapped.device)

    def _build_action_preprocessor(self, env: DirectRLEnv) -> RunningStandardScaler:
        """Maps the actor's raw (unbounded) output to physical joint targets."""
        soft_joint_pos_limits = env.unwrapped.robot.data.soft_joint_pos_limits.torch
        dof_lower_limits = soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = soft_joint_pos_limits[0, :, 1]
        action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        action_scale = dof_upper_limits - dof_lower_limits

        return RunningStandardScaler(
            size=action_offset.shape[-1],
            mean=action_offset,
            variance=action_scale ** 2,
        ).to(env.unwrapped.device)

    @abstractmethod
    def _initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        pass

    @abstractmethod
    def _initialize_optimizer(self) -> None:
        pass

    @abstractmethod
    def _initialize_storage(self, env: DirectRLEnv) -> None:
        pass

    @abstractmethod
    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample actions for the current observations. """
        pass

    @abstractmethod
    def process_env_step(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        """Record reward, done flags, and optional step info, then flush the
        current transition into the rollout buffer. """
        self._track_episode_stats(rewards, terminated | truncated, infos or {})

    def _track_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict) -> None:
        """Accumulate per-env episode reward/length, flush completed episodes into the
        rolling trackers, and push the resulting stats/tags to the logger."""
        timestep = self.logger.timestep

        if self._cumulative_rewards is None:
            self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

        self._cumulative_rewards.add_(rewards)
        self._cumulative_timesteps.add_(1)

        if dones.any():
            self._track_rewards.extend(self._cumulative_rewards[dones].tolist())
            self._track_timesteps.extend(self._cumulative_timesteps[dones].tolist())

            self._cumulative_rewards[dones] = 0
            self._cumulative_timesteps[dones] = 0

        if len(self._track_rewards):
            track_rewards = np.array(self._track_rewards)
            track_timesteps = np.array(self._track_timesteps)

            self.mean_episode_reward = float(np.mean(track_rewards))

            self.logger.track_data("Performance/Episode Reward (max)", np.max(track_rewards), timestep)
            self.logger.track_data("Performance/Episode Reward (min)", np.min(track_rewards), timestep)
            self.logger.track_data("Performance/Episode Reward (mean)", self.mean_episode_reward, timestep)

            self.logger.track_data("Performance/Episode Length (max)", np.max(track_timesteps), timestep)
            self.logger.track_data("Performance/Episode Length (min)", np.min(track_timesteps), timestep)
            self.logger.track_data("Performance/Episode Length (mean)", np.mean(track_timesteps), timestep)

            core_performance = {
                "Episode Reward (mean)": self.mean_episode_reward,
                "Episode Reward (max)": float(np.max(track_rewards)),
                "Episode Reward (min)": float(np.min(track_rewards)),
                "Episode Length (mean)": float(np.mean(track_timesteps)),
            }

            mean_episode_time = float(np.mean(track_timesteps) * self._step_dt)
            self.logger.track_data("Performance/Episode Time (mean) [s]", mean_episode_time, timestep)
            core_performance["Episode Time (mean) [s]"] = mean_episode_time

            self.logger.set_core_performance(core_performance)

        for k, v in infos.get("log", {}).items():
            self.logger.track_data(tag=k, value=v, step=timestep)

        motion_ids = infos.get("motion_ids")
        for name, values in infos.get("reward_terms", {}).items():
            self._accumulate_reward_term(name, values, motion_ids)

        core_rewards = {}
        for key, count in self._reward_term_count.items():
            if count:
                mean_value = self._reward_term_sum[key] / count
                self.logger.track_data(f"Reward/{key} (mean)", mean_value, timestep)
                core_rewards[f"{key} (mean)"] = mean_value

        if core_rewards:
            self.logger.set_core_rewards(core_rewards)

    def _accumulate_reward_term(
        self,
        name: str,
        values: torch.Tensor,
        motion_ids: torch.Tensor | None = None,
    ) -> None:
        """Credit one step's per-env values for reward term `name` (e.g. "task",
        "style") into its running mean over the whole training run. When `motion_ids`
        is given, also credited into a per-clip running mean (e.g. "task/walk")."""
        self._reward_term_sum[name] += values.sum().item()
        self._reward_term_count[name] += values.numel()

        if motion_ids is None:
            return

        for motion_id, motion_name in enumerate(self._motion_names):
            mask = motion_ids == motion_id
            if mask.any():
                key = f"{name}/{motion_name}"
                self._reward_term_sum[key] += values[mask].sum().item()
                self._reward_term_count[key] += int(mask.sum().item())

    @abstractmethod
    def update(self) -> None:
        """Run gradient updates."""
        self._update_step += 1

    @abstractmethod
    def inference(self) -> None:
        """Inference Mode """

    @abstractmethod
    def train(self) -> None:
        """Train mode"""

    @abstractmethod
    def write_checkpoint(self, timestep: int, filename: str | None = None) -> None:
        """Save the agent's models to the specified path."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str, device: torch.device) -> None:
        """Load the agent's models from the specified path."""
        pass
