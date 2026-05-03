from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from moveitmoveit.src.buffers import CircularObsBuffer
from moveitmoveit.src.models import AMPNetworks
from moveitmoveit.src.models.norm import EmpiricalNorm
from utils import Logger

from .ppo import PPO, PPOHyperparams

@dataclass(frozen=True)
class AMPHyperparams(PPOHyperparams):
    # Discriminator replay-buffer settings
    discriminator_buffer_capacity: int = 100_000

    # Discriminator optimizer
    disc_lr: float = 1e-4

    # How often (in policy update iterations) to update the discriminator
    discriminator_update_interval: int = 4

    # Number of gradient steps per discriminator update
    disc_update_epochs: int = 1
    disc_mini_batches: int = 4

    disc_grad_penalty_coef: float = 5.0
    num_disc_obs_steps: int = 10

    style_reward_lambda: float = 0.5 
    goal_reward_lambda: float = 0.5

class AMP(PPO):
    """Adversarial Motion Priors (AMP) algorithm. """

    def __init__(
        self,
        networks: AMPNetworks,
        params: AMPHyperparams,
        logger: Logger,
    ):  
        super().__init__(networks, params, logger)

        self._update_count = 0

    def init_storage(
        self,
        num_envs: int,
        num_transitions: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        super().init_storage(num_envs, num_transitions, obs_dim, action_dim)

        self.discriminator_storage = CircularObsBuffer(
            obs_dim=obs_dim,
            disc_obs_steps=self.params.num_disc_obs_steps,
            n_envs=num_envs,
            capacity=self.params.discriminator_buffer_capacity,
            device=self.networks.device
        )

    def process_reset(self, infos: dict, env_ids = None) -> None:
        """Seed the discriminator sliding window from motion-lib frames returned at reset."""
        self.discriminator_storage.seed_from_windows(infos["disc_obs"], env_ids=env_ids)

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        # Slide the window left by one and append the new sim observation,
        # then flush every env's window to the ring buffer.
        self.discriminator_storage.add(infos["disc_obs"])

        super().process_env_step(rewards, dones, infos)

    def update(self, optimizer: torch.optim.Optimizer) -> None:
        # PPO policy update
        super().update(optimizer)

        self._update_count += 1

        # Periodically update the discriminator
        if self._update_count % self.params.discriminator_update_interval == 0:
            self._update_discriminator()

    def _update_discriminator(self) -> None:
        """Run one round of discriminator gradient updates."""
        if self.discriminator_storage.size == 0:
            return

        mean_disc_loss = 0.0
        num_updates = self.params.disc_update_epochs * self.params.disc_mini_batches

        agent_batch_size = max(
            1,
            self.discriminator_storage.size // self.params.disc_mini_batches,
        )

        for _ in self.discriminator_storage.mini_batch_generator(
            num_mini_batches=self.params.disc_mini_batches,
            num_epochs=self.params.disc_update_epochs,
            sample_size=agent_batch_size * self.params.disc_mini_batches,
        ):
            agent_obs, agent_next_obs = _

            # Sample an equal-sized batch of reference motion transitions
            ref_obs, ref_next_obs = self.ref_obs_sampler(len(agent_obs))

            # Concatenate (s, s') as the discriminator input
            agent_input = torch.cat([agent_obs, agent_next_obs], dim=-1)
            ref_input = torch.cat([ref_obs, ref_next_obs], dim=-1)

            # Logistic regression: agent → 0, reference → 1
            agent_logits = self.networks.disc(agent_input)
            ref_logits = self.networks.disc(ref_input)

            agent_loss = nn.functional.binary_cross_entropy_with_logits(
                agent_logits, torch.zeros_like(agent_logits)
            )
            ref_loss = nn.functional.binary_cross_entropy_with_logits(
                ref_logits, torch.ones_like(ref_logits)
            )
            disc_loss = agent_loss + ref_loss

            # gradient penalty on reference inputs
            if self.params.disc_grad_penalty_coef > 0:
                inputs = ref_input.detach().requires_grad_(True)
                outputs = self.networks.discriminator(inputs)
                grads = torch.autograd.grad(
                    outputs=outputs.sum(),
                    inputs=inputs,
                    create_graph=True,
                )[0]
                grad_pentalty = grads.pow(2).sum(dim=-1).mean()
                disc_loss = disc_loss + self.params.disc_grad_penalty_coef * grad_pentalty

            self.disc_optimizer.zero_grad()
            disc_loss.backward()
            self.disc_optimizer.step()

            mean_disc_loss += disc_loss.item()

        mean_disc_loss /= num_updates
        self.logger.log_metric("disc/loss", mean_disc_loss)


        # stacked = torch.concatenate([prev_disc_obs, disc_obs], dim=1)
        #un problemo here 
        # with torch.no_grad():
        #     d = self.networks.disc(stacked).squeeze()
        
        # style_reward = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0).clone()
        # goal_reward = rewards.clone()
        # rewards = self.params.style_reward_lambda * style_reward + self.params.goal_reward_lambda * goal_reward
        # self.discriminator_storage.add(disc_obs)