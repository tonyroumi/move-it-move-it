from __future__ import annotations

import collections
import itertools
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from isaaclab.envs import DirectRLEnv

from skrl.resources.preprocessors.torch import RunningStandardScaler

from moveitmoveit.models import GaussianMLP, MLP
from moveitmoveit.storage import RolloutStorage
from moveitmoveit.utils.logger import Logger

from .ppo_cfg import PPOCfg
from ..base import BaseAgent

def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    last_values: torch.Tensor,
    dones: torch.Tensor,
    gae_lambda: float,
    discount: float,
)-> torch.Tensor:

    advantage = 0
    advantages = torch.zeros_like(rewards)
    steps = rewards.shape[0]

    for i in reversed(range(steps)):
        next_values = values[i+1] if i < steps - 1 else last_values

        advantage = (
            rewards[i] - values[i] + discount * (1 - dones[i]) * (next_values + gae_lambda * advantage)
        )

        advantages[i] = advantage

    gae_returns = advantages + values

    return advantages, gae_returns

class PPO(BaseAgent):
    cfg: PPOCfg

    def __init__(self, cfg: dict, logger: Logger):
        super().__init__(cfg=cfg, logger=logger)

        self.grad_step = 0

    def _initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
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

        self._obs_preprocessor = RunningStandardScaler(size=obs_size).to(env.unwrapped.device)
        self._value_preprocessor = RunningStandardScaler(size=1).to(env.unwrapped.device)

    def _initialize_storage(
        self,
        env: DirectRLEnv,
        storage_cfg: dict,
    ) -> None:
        self.storage = RolloutStorage(
            num_envs=env.unwrapped.num_envs,
            num_transitions_per_env=self.cfg.num_transitions_per_env,
            obs_dim=env.observation_space.shape[-1],
            action_dim=env.action_space.shape[-1],
            device=env.unwrapped.device
        )
        self.transition = self.storage.Transition()

        self._next_observations = None

    def _initialize_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            itertools.chain(self.actor.parameters(), self.critic.parameters()),
            lr=self.cfg.learning_rate,
        )

    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        normed_obs = self._obs_preprocessor(observations)
        
        actions = self.actor(normed_obs, deterministic=deterministic)
        values = self.critic(normed_obs)

        self.transition.observations = observations
        self.transition.actions = actions # sampled action
        self.transition.actions_log_prob = self.actor.get_actions_log_prob(actions) #log_prob
        self.transition.values = self._value_preprocessor(values, inverse=True)

        return actions

    def process_env_step(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        self.transition.rewards = rewards
        self.transition.dones = terminated

        self.storage.add_transition(self.transition)
        self.transition.clear()

        self._next_observations = next_observations

    def update(self) -> None:
        super().update()

        with torch.no_grad():
            last_observations = self._obs_preprocessor(self._next_observations)
            last_values = self.critic(last_observations)
            last_values = self._value_preprocessor(last_values, inverse=True)

        advantages, returns = compute_gae(
            self.storage.rewards,
            self.storage.values,
            last_values,
            self.storage.dones,
            self.cfg.gae_lambda,
            self.cfg.discount,
        )
        self._diagnostics["Iter/Advantage"] = advantages.tolist()

        if (not self.cfg.normalize_advantage_per_mini_batch):
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        self.storage.set_advantage(advantages)
        self.storage.set_returns(self._value_preprocessor(returns, train=True))

        for epoch in range(self.cfg.num_learning_epochs):
            for batch in self.storage.mini_batch_generator(self.cfg.num_mini_batches):

                if (self.cfg.normalize_advantage_per_mini_batch):
                    advantage = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)
                else:
                    advantage = batch.advantages

                # update normalizer with samples that we update the policy from once. 
                observations = self._obs_preprocessor(batch.observations, train=(not epoch))

                # update policy distribution
                self.actor(observations)
                actions_log_prob = self.actor.get_actions_log_prob(batch.actions).unsqueeze(-1)

                # compute approximate KL divergence
                with torch.no_grad():
                    log_ratio = actions_log_prob - batch.old_actions_log_prob
                    kl_divergence = ((torch.exp(log_ratio) - 1) - log_ratio).mean()
                    # kl_divergences.append(kl_divergence)

                ratio = torch.exp(actions_log_prob - batch.old_actions_log_prob)
                surrogate = advantage * ratio 
                surrogate_clipped = advantage * torch.clip(
                            ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param
                )

                policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                values = self.critic(observations)

                # For Diagnostics
                with torch.no_grad():
                    explained_variance = (
                        1.0
                        - (batch.returns - values).var()
                        / (batch.returns.var() + 1e-8)
                    )
                    clip_fraction = (
                        (torch.abs(ratio - 1.0) > self.cfg.clip_param)
                        .float()
                        .mean()
                    )

                if self.cfg.use_clipped_value_loss:
                    values = batch.values + torch.clip(
                        values - batch.values, -self.cfg.value_loss_clip_param, self.cfg.value_loss_clip_param
                    )
                value_loss = self.cfg.value_loss_coef * F.mse_loss(values, batch.returns)

                loss = policy_loss + value_loss #- self.cfg.entropy_coef * self.actor.entropy

                self.optimizer.zero_grad()
                loss.backward()

                self._diagnostics["Policy Grad Norm"].append(self.actor.grad_norm)
                self._diagnostics["Value Grad Norm"].append(self.critic.grad_norm)

                theta_before = torch.cat([
                    p.detach().flatten().clone()
                    for p in self.actor.parameters()
                ])

                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(),
                    max_norm=self.cfg.max_grad_norm,
                )

                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(),
                    max_norm=self.cfg.max_grad_norm,
                )

                self.optimizer.step()

                theta_after = torch.cat([
                    p.detach().flatten()
                    for p in self.actor.parameters()
                ])

                relative_update = (
                    torch.norm(theta_after - theta_before)
                    / torch.norm(theta_before)
                )

                self._diagnostics["Relative Policy Update"].append(relative_update)

                # Diagnostics
                self._diagnostics["Total Loss"].append(loss.item())
                self._diagnostics["Policy Loss"].append(policy_loss.item())
                self._diagnostics["Value Loss"].append(value_loss.item())
                self._diagnostics["KL Divergence"].append(kl_divergence.item())
                self._diagnostics["Clip Fraction"].append(clip_fraction.item())
                self._diagnostics["Clip Ratio (std)"].append(ratio.std().item())
                self._diagnostics["Clip Ratio (min)"].append(ratio.min().item())
                self._diagnostics["Clip Ratio (max)"].append(ratio.max().item())
                self._diagnostics["Explained Variance"].append(explained_variance.mean().item())

                self.grad_step += 1

        self.storage.clear()

        diagnostics = self._diagnostics
        self._diagnostics = collections.defaultdict(list)
        return self.grad_step, diagnostics

    def write_checkpoint(self, timestep: int) -> None:
        """Save the agent's models to the specified path."""
        path = os.path.join(self.logger.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)

        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "_obs_preprocessor": self._obs_preprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
        }, f"{path}/{timestep}.pt")

    def load_checkpoint(self, path: str, device: torch.device) -> None:
        """Load the agent's models from the specified path."""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self._obs_preprocessor.load_state_dict(checkpoint["_obs_preprocessor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
