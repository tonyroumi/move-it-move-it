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

#one of observations 
# one for dics observations. these are separate. makes sense


# we will also need an action normalizer. what they do is the 
# this is what they do in mimickit
#            a_mean = torch.tensor(0.5 * (a_space.high + a_space.low), device=self._device, dtype=a_dtype)
#            a_std = torch.tensor(0.5 * (a_space.high - a_space.low), device=self._device, dtype=a_dtype)
#
# assert (a_std > 0).all().item(), "init_std must be > 0 for action normalizer (Box action space wrong! Check your XML file. Joints must have 'limited=true' and non-zero bounds.)"

# a_norm = normalizer.Normalizer(a_mean.shape, device=self._device, init_mean=a_mean, 
#                                      init_std=a_std, dtype=a_dtype)

# so why do they do this?:
# well... 
# we can model the standard deviation (average spread of values around the mean) with some basic assumptions about the type of distribution that it is
# for a uniform distribution (each value has an equal chance of being selected in a range) we can model the standard deviation
# of a uniform random variable whose support is exactly [a,b] through b-a/sqrt(12). prob derivation. just trust.
# 

# they use action normalizer to unormalize the action the model produces.
# why?

#they update normalizers each env step (32 times) and post update step.
# each env step they call record(). update step they call update.

#dicriminator optimizers and agent optimizers are the same dafuq

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

    style_reward_lambda: float = 0.5 
    goal_reward_lambda: float = 0.5

class AMP(PPO):
    """Adversarial Motion Priors (AMP) algorithm. """

    def __init__(
        self,
        networks: AMPNetworks,
        params: AMPHyperparams = AMPHyperparams(),
        logger: Logger = Logger(),
    ):
        super().__init__(networks, params, logger)

        self.ref_obs_sampler = None

        self._update_count = 0

        self.disc_optimizer = optim.Adam(
            self.networks.discriminator.parameters(),
            lr=params.disc_lr,
        )

    def init_storage(
        self,
        num_envs: int,
        num_transitions: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        super().init_storage(num_envs, num_transitions, obs_dim, action_dim)

        disc_obs_dim = self.networks.discriminator.in_channels
        self.discriminator_storage = CircularObsBuffer(
            capacity=self.params.discriminator_buffer_capacity,
            num_envs=num_envs,
            obs_dim=disc_obs_dim,
            device=self.networks.device,
        )

    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        prev_disc_obs = torch.as_tensor(infos["prev_disc_obs"], dtype=torch.float32, device=self.networks.device)
        disc_obs = torch.as_tensor(infos["disc_obs"], dtype=torch.float32, device=self.networks.device)

        stacked = torch.concatenate([prev_disc_obs, disc_obs], dim=1)
        #un problemo here 
        with torch.no_grad():
            d = self.networks.disc(stacked).squeeze()
        
        style_reward = torch.clamp(1.0 - 0.25 * (d - 1.0) ** 2, min=0.0).clone()
        goal_reward = rewards.clone()
        rewards = self.params.style_reward_lambda * style_reward + self.params.goal_reward_lambda * goal_reward

        super().process_env_step(rewards, dones, infos)

        self.discriminator_storage.add(stacked)
        self.networks.record_disc_obs(stacked)

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