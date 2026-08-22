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

def compute_gae(
    num_transitions_per_env: int,
    rewards: torch.Tensor,
    values: torch.Tensor,
    dones: torch.Tensor,
    gae_lambda: float,
    discount: float
)-> torch.Tensor:
    advantages = torch.zeros_like(rewards)
    for i in reversed(range(num_transitions_per_env-2, -1)):
        td_error = (
            rewards[i] + discount * (1 - dones[i]) * values[i+1] - values[i]
        )
        advantages[i] = td_error + gae_lambda * discount * advantages[i+1]

    return advantages

class PPO(BaseAgent):
    cfg: PPOCfg

    def __init__(self, cfg: dict, logger: Logger):
        super().__init__(cfg=cfg, logger=logger)

    def initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        obs_size = env.observation_space.shape[-1]

        self.actor = GaussianMLP(
            in_channels=obs_size,
            out_channels=env.action_space.shape[-1],
            **model_cfg["actor"]
        ).to(env.unwrapped.device)
        self.critic = MLP(
            in_channels=obs_size,
            out_channels=1,
            **model_cfg["critic"]
        ).to(env.unwrapped.device)

        self.obs_preproccessor = RunningStandardScaler(size=obs_size).to(env.unwrapped.device)

    def initialize_storage(
        self,
        env: DirectRLEnv,
        num_transitions_per_env: int,
        storage_cfg: dict,
    ) -> None:
        self.storage = RolloutStorage(
            num_envs=env.unwrapped.num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs_dim=env.observation_space.shape[-1],
            action_dim=env.action_space.shape[-1],
            device=env.unwrapped.device
        )
        self.transition = self.storage.Transition()

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        result = {
            "observations": self.obs_preproccessor(observations, train=(not deterministic)),
            "value": self.critic(observations)
        }
        actions = self.actor(result["observations"], deterministic=deterministic)

        self.transition.observations = observations
        self.transition.actions = actions # sampled action
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(actions) #log_prob
        self.transition.values = result["value"]

        return actions
    
    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        timeout: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        self.transition.rewards = rewards
        self.transition.dones = terminated

        self.storage.add_transition(self.transition)

    def update(self, optimizer: torch.optim.Optimizer) -> None:
        metrics = {}

        advantages = compute_gae(
            self.storage.num_transitions_per_env,
            self.storage.rewards,
            self.storage.values,
            self.storage.dones,
            self.cfg.gae_lambda,
            self.cfg.discount
        )

        if (not self.cfg.normalize_advantage_per_mini_batch):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.storage.advantages = advantages

        for batch in self.storage.mini_batch_generator(
            self.cfg.num_mini_batches,
            self.cfg.num_learning_epochs
        ):
            if (self.cfg.normalize_advantage_per_mini_batch):
                advantage = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)

            observations = self.obs_preproccessor(batch.observations)
            
            # update policy distribution
            self.actor(observations)

            # actor loss
            actions_log_prob = self.actor.get_actions_log_prob(batch.actions)
            surrogate = advantage * torch.exp(actions_log_prob - batch.old_actions_log_prob)
            surrogate_clipped = surrogate.clip(1 - self.cfg.clip_param, 1 + self.cfg.clip_param)

            # actor loss
            actor_loss = -surrogate_clipped
            # critic loss
            value_loss = torch.mean(torch.pow(batch.returns - self.critic(observations)))

            if self.cfg.use_clipped_value_loss:
                value_loss = value_loss.clip(1 - self.cfg.value_loss_clip_param, 1 + self.cfg.value_loss_clip_param)

            loss = actor_loss + value_loss * self.cfg.value_loss_coef - self.cfg.entropy_coef * self.actor.entropy

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                max_norm=self.cfg.max_grad_norm,
            )

            old_dist = self.actor.dist

            optimizer.step()

            self.grad_step += 1

            new_dist = self.actor.dist

            kl = torch.distributions.kl_divergence(
                old_dist, new_dist
            ).sum(dim=-1).mean()

            metrics.update({
                "train/actor_loss": actor_loss,
                "train/value_loss": value_loss,
                "train/kl_divergence": kl,
                "train/entropy": self.actor.entropy,
                "train/advantage_mean": advantages.mean(),
                "train/advantage_std": advantages.std(),
                "train/value_estimate_mean": batch.values.mean(),
                "train/value_estimate_std": batch.values.std(),
            })

            self.logger.log_scalars(metrics, self.grad_step)

        self.storage.clear()
