from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import collections
import os
import time

import numpy as np
import torch
import torch.optim as optim
import tqdm

from isaaclab.envs import DirectRLEnv

from moveitmoveit.agents import resolve_agent
from moveitmoveit.utils.logger import Logger

@dataclass(kw_only=True)
class OnPolicyRunnerCfg:
    timesteps: int = 100000
    """Number of timesteps to train/evaluate for."""

    write_interval: int = 1000
    """ Interval to log shtuff. """

    checkpoint_interval: int = -1
    """ Interval to save checkpoints. -1 Implies save based on test performance. """

class OnPolicyRunner:
    """Generic on-policy training loop. """

    def __init__(
        self,
        *,
        cfg: dict,
        env: DirectRLEnv,
        logger: Logger,
    ):
        self.cfg = OnPolicyRunnerCfg(**cfg["runner"])
        self.env = env
        self.logger = logger

        # Diagnostics
        self._track_rewards = collections.deque(maxlen=100)
        self._track_timesteps = collections.deque(maxlen=100)
        self._cumulative_rewards = None
        self._cumulative_timesteps = None
        self._mean_episode_reward = None
        self._best_mean_episode_reward = float("-inf")

        # Agent
        self._initialize_agent(cfg)

    def _initialize_agent(self, cfg: dict):
        agent_cls, cfg_cls = resolve_agent(cfg["agent"]["class_type"])

        self.agent = agent_cls(
            cfg = cfg_cls(**cfg["agent"]),
            logger = self.logger
        )
        self.agent.init(self.env, cfg)

    def learn(self) -> None:
        observations, infos = self.env.reset()

        total_iterations = self.cfg.timesteps // self.agent.cfg.num_transitions_per_env
        timestep = 0

        for iteration in tqdm.tqdm(range(total_iterations), disable=False):
            for _ in range(self.agent.cfg.num_transitions_per_env):
                with torch.no_grad():
                    actions = self.agent.act(observations)

                    observations, rewards, terminated, truncated, infos = self.env.step(actions)
                    timestep += 1

                    self.agent.process_env_step(
                        next_observations=observations,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=truncated,
                        infos=infos
                    )

                    dones = terminated | truncated
                    self.write_env_diagnostics(rewards, dones, infos, timestep)

            grad_steps, infos = self.agent.update()
            self.write_agent_diagnostics(infos, iteration, grad_steps, timestep)

            if iteration % self.cfg.checkpoint_interval == 0:
                self.agent.write_checkpoint(timestep)
            if self._mean_episode_reward is not None and self._mean_episode_reward > self._best_mean_episode_reward:
                self._best_mean_episode_reward = self._mean_episode_reward
                self.agent.write_checkpoint(timestep, filename="best_agent.pt")
            self.logger.write_data()

    def write_env_diagnostics(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict, timestep):
        if self._cumulative_rewards is None:
            self._cumulative_rewards = torch.zeros_like(rewards, dtype=torch.float32)
            self._cumulative_timesteps = torch.zeros_like(rewards, dtype=torch.int32)

        self._cumulative_rewards.add_(rewards)
        self._cumulative_timesteps.add_(1)

        if dones.any():
            self._track_rewards.extend(self._cumulative_rewards[dones].tolist())
            self._track_timesteps.extend(self._cumulative_timesteps[dones].tolist())

            # reset the cumulative rewards and timesteps
            self._cumulative_rewards[dones] = 0
            self._cumulative_timesteps[dones] = 0

        if len(self._track_rewards):
            track_rewards = np.array(self._track_rewards)
            track_timesteps = np.array(self._track_timesteps)

            self._mean_episode_reward = np.mean(track_rewards)

            self.logger.track_data("Performance/Episode Reward (max)", np.max(track_rewards), timestep)
            self.logger.track_data("Performance/Episode Reward (min)", np.min(track_rewards), timestep)
            self.logger.track_data("Performance/Episode Reward (mean)", self._mean_episode_reward, timestep)

            self.logger.track_data("Performance/Episode Length (max)", np.max(track_timesteps), timestep)
            self.logger.track_data("Performance/Episode Length (min)", np.min(track_timesteps), timestep)
            self.logger.track_data("Performance/Episode Length (mean)", np.mean(track_timesteps), timestep)

            self.logger.track_data(
                "Performance/Episode Time (mean) [s]", np.mean(track_timesteps) * self.env.step_dt, timestep
            )

        for k, v in infos.get("log", {}).items():
            self.logger.track_data(tag=k, value=v, step=timestep)

    def write_agent_diagnostics(self, infos: dict[str, list], iteration: int, grad_step: int, timestep: int):
        for k, v in infos.items():
            if "Iter" in k:
                k = k.split("/")[-1]
                self.logger.track_data(f"Debug /{k} (mean)", np.mean(v[0]), iteration)
            else:
                # per-minibatch values collected across the whole update; report their mean
                value = sum(v) / len(v) if isinstance(v, list) else v
                self.logger.track_data(f"Train/{k}", value, grad_step)
