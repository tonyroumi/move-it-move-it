from __future__ import annotations

import math
from typing import Any, Dict

import torch
import torch.nn as nn

from isaaclab.envs import DirectRLEnv

from skrl.resources.preprocessors.torch import RunningStandardScaler

from moveitmoveit.models import GaussianMLP, MLP
from moveitmoveit.storage import RolloutStorage
from moveitmoveit.utils.logger import Logger

from .ppo_cfg import PPOCfg
from ..base import BaseAgent

class PPO(BaseAgent):
    cfg: PPOCfg

    def __init__(self, cfg: PPOCfg, logger: Logger):
        super().__init__(cfg=cfg, logger=logger)

    def initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        obs_size = env.single_observation_space.shape[0]

        self.actor = GaussianMLP(
            in_channels=obs_size,
            out_channels=env.single_action_space.shape[0],
            **model_cfg["actor"]
        ).to(env.device)
        self.critic = MLP(
            in_channels=obs_size,
            out_channels=1,
            **model_cfg["critic"]
        ).to(env.device)

        self.obs_preproccessor = RunningStandardScaler(size=obs_size)

    def initialize_storage(
        self,
        env: DirectRLEnv,
        num_transitions_per_env: int,
        **kwargs: Any
    ) -> None:
        self.storage = RolloutStorage(
            num_steps=num_transitions_per_env,
            num_envs=env.num_envs,
            obs_shape=env.single_observation_space.shape,
            action_shape=env.single_action_space.shape,
            device=env.device
        )
        self.transition = self.storage.Transition()

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        # We pick back up here... 
        values = self.evaluate(observations)
        self.transition.observations = observations

        observations = self.obs_preproccessor(observations)
        actions = self.actor(observations, deterministic=deterministic)

        self.transition.actions = actions
        self.transition.values = self.evaluate(observations)

        return actions

    def evaluate(self, observations: torch.Tensor) -> torch.Tensor:
        values = self.critic(observations)

        if actions is None:
            actions = self.transition.actions

        log_probs, entropy = self.actor.evaluate_log_prob(observations, actions)

        return values, log_probs, entropy

    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        pass

    def compute_returns(self, last_values: torch.Tensor) -> None:
        pass

    def update(self, optimizer: torch.optim.Optimizer) -> None:
        super().update(optimizer)
