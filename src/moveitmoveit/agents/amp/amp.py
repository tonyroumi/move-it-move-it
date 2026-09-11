from __future__ import annotations

from typing import Callable
import itertools
import os

import torch
import torch.nn as nn

from isaaclab.envs import DirectRLEnv

from skrl.resources.preprocessors.torch import RunningStandardScaler

from moveitmoveit.models import MLP
from moveitmoveit.storage import TensorCircularBuffer
from moveitmoveit.utils.logger import Logger

from .amp_cfg import AMPCfg
from ..ppo import PPO


class AMP(PPO):
    """Adversarial Motion Priors (AMP) algorithm. """
    cfg: AMPCfg

    def __init__(self, cfg: AMPCfg, logger: Logger):
        super().__init__(cfg=cfg, logger=logger)

    def init(self, env: DirectRLEnv, cfg: dict):
        super().init(env, cfg)

        self.collect_reference_motions: Callable = env.unwrapped.collect_reference_motions
        self._sample_reference_clip_times: Callable = env.unwrapped.sample_reference_clip_times
        self._motion_dt: float = env.unwrapped.motion_dt

    def _initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        super()._initialize_models(env, model_cfg)

        # single-frame AMP feature width (never includes joystick commands, since a reference
        # motion clip has no notion of a commanded velocity)
        self._amp_single_obs_dim = env.unwrapped.collect_reference_motions(1).shape[-1]
        self.amp_obs_dim = self.cfg.num_amp_observations * self._amp_single_obs_dim

        self.discriminator = MLP(
            in_channels=self.amp_obs_dim,
            out_channels=1,
            **model_cfg["discriminator"]
        ).to(env.unwrapped.device)

        self._disc_obs_preprocessor = RunningStandardScaler(size=self.amp_obs_dim).to(env.unwrapped.device)

    def _initialize_optimizer(self) -> None:
        super()._initialize_optimizer()
        self.disc_optimizer = torch.optim.Adam(
            itertools.chain(
                self.discriminator.parameters()
            ),
            lr=self.cfg.disc_lr,
        )

    def _initialize_storage(
        self,
        env: DirectRLEnv,
        storage_cfg: dict,
    ) -> None:
        super()._initialize_storage(env, storage_cfg)

        self.buf_capacity = storage_cfg.get("capacity", 2_000_000)

        self._amp_observations_buf = None
        # rolling per-env window of the last `num_amp_observations` raw (single-frame, command-free)
        # observations, newest first; lazily initialized on the first env step
        self._live_amp_history = None

    def _update_live_amp_history(self, next_observations: torch.Tensor, done_mask: torch.Tensor) -> torch.Tensor:
        raw_frame = next_observations[..., : self._amp_single_obs_dim]

        if self._live_amp_history is None:
            # cold start: seed every env's window with its first observation, as if just reset
            self._live_amp_history = raw_frame.unsqueeze(1).repeat(1, self.cfg.num_amp_observations, 1)
        else:
            self._live_amp_history = torch.roll(self._live_amp_history, shifts=1, dims=1)
            self._live_amp_history[:, 0] = raw_frame
            if done_mask.any():
                # envs that reset this step start a new episode; don't mix in pre-reset history
                self._live_amp_history[done_mask] = raw_frame[done_mask].unsqueeze(1)

        return self._live_amp_history.view(next_observations.shape[0], -1)

    def process_env_step(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        super().process_env_step(next_observations, rewards, terminated, truncated, infos)

        amp_obs = self._update_live_amp_history(next_observations, terminated | truncated)

        # Lazy load motion buffers after first step
        if self._amp_observations_buf is None:
            self._amp_observations_buf = TensorCircularBuffer(
                capacity=self.buf_capacity,
                sample_shape=(amp_obs.shape[-1],),
                device=rewards.device,
            )

        self._amp_observations_buf.append(amp_obs)

    def update(self) -> None:
        rewards = self.storage.rewards
        amp_observations = self._amp_observations_buf.get_since_last()

        with torch.no_grad():
            disc_logits = self.discriminator(
                self._disc_obs_preprocessor(amp_observations)
            )
            style_reward = -torch.log(
                torch.maximum(1 - 1 / (1 + torch.exp(-disc_logits)), torch.tensor(0.0001, device=rewards.device))
            ).view(rewards.shape)

        scaled_style_reward = self.cfg.style_reward_lambda * style_reward
        combined_rewards = self.cfg.goal_reward_lambda * rewards + scaled_style_reward
        rewards.copy_(combined_rewards)

        # Style reward is logged against policy update nums
        self.logger.add_info("Style Reward", scaled_style_reward.mean().item())
        # Discriminator prediction logits is against discriminator updates
        self.logger.add_info("Disc Prediction Logits", disc_logits.mean().item(), 2)

        super().update()

        if self._update_step % self.cfg.discriminator_update_interval == 0:
            self._update_discriminator()

    def _sample_reference_motion_window(self, num_samples: int) -> torch.Tensor:
        """`num_amp_observations` consecutive reference-motion frames per sample, newest first,
        concatenated to match the layout of the live rollout's stacked AMP observations."""
        clip_indexes, times = self._sample_reference_clip_times(num_samples)
        frames = [
            self.collect_reference_motions(
                num_samples, current_times=times - k * self._motion_dt, clip_indexes=clip_indexes
            )
            for k in range(self.cfg.num_amp_observations)
        ]
        return torch.cat(frames, dim=-1)

    def _update_discriminator(self) -> None:
        """Run one round of discriminator gradient updates."""
        for _ in range(self.cfg.disc_num_updates):
            ref_motion = self._sample_reference_motion_window(self.cfg.disc_batch_size)
            agent_motion = self._amp_observations_buf.sample(self.cfg.disc_batch_size)

            ref_motion = self._disc_obs_preprocessor(ref_motion, train=True)
            agent_motion = self._disc_obs_preprocessor(agent_motion, train=True)

            ref_motion.requires_grad_(True)
            agent_logits = self.discriminator(agent_motion)
            ref_logits = self.discriminator(ref_motion)

            discriminator_loss = 0.5 * (
                    nn.BCEWithLogitsLoss()(agent_logits, torch.zeros_like(agent_logits))
                    + torch.nn.BCEWithLogitsLoss()(ref_logits, torch.ones_like(ref_logits))
                )

            if self.cfg.disc_logit_reg:
                logit_weights = torch.flatten(self.discriminator.get_logit_weights())
                discriminator_loss += self.cfg.disc_logit_reg * torch.sum(
                    torch.square(logit_weights)
                )

            if self.cfg.disc_grad_penalty:
                amp_motion_gradient = torch.autograd.grad(
                    ref_logits,
                    ref_motion,
                    grad_outputs=torch.ones_like(ref_logits),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )
                gradient_penalty = torch.sum(torch.square(amp_motion_gradient[0]), dim=-1).mean()
                discriminator_loss += self.cfg.disc_grad_penalty * gradient_penalty

            # discriminator weight decay
            if self.cfg.disc_weight_decay:
                weights = [
                    torch.flatten(module.weight)
                    for module in self.discriminator.modules()
                    if isinstance(module, torch.nn.Linear)
                ]
                weight_decay = torch.sum(torch.square(torch.cat(weights, dim=-1)))
                discriminator_loss += self.cfg.disc_weight_decay * weight_decay

            discriminator_loss *= self.cfg.disc_loss_scale

            self.disc_optimizer.zero_grad()
            discriminator_loss.backward()

            if self.cfg.grad_norm_clip > 0:
                nn.utils.clip_grad_norm_(
                    itertools.chain(
                        self.discriminator.parameters()
                    ),
                    self.cfg.grad_norm_clip,
                )

            self.disc_optimizer.step()

            self.logger.add_info("Discriminator Loss", discriminator_loss.item(), 2)
            self.logger.add_info("Agent Motion Logits", agent_logits.mean().item(), 2)
            self.logger.add_info("Reference Motion Logits", ref_logits.mean().item(), 2)
            self.logger.step_metric(2)

    def write_checkpoint(self, timestep: int, filename: str | None = None) -> None:
        """Save the agent's models to the specified path."""
        path = os.path.join(self.logger.log_dir, "checkpoints")
        os.makedirs(path, exist_ok=True)

        filename = filename if filename is not None else f"{timestep}.pt"
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "obs_preprocessor": self._obs_preprocessor.state_dict(),
            "disc_obs_preprocessor": self._disc_obs_preprocessor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "disc_optimizer": self.disc_optimizer.state_dict(),
        }, f"{path}/{filename}")

    def load_checkpoint(self, path: str, device: torch.device) -> None:
        """Load the agent's models from the specified path."""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self._obs_preprocessor.load_state_dict(checkpoint["obs_preprocessor"])
        self._disc_obs_preprocessor.load_state_dict(checkpoint["disc_obs_preprocessor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
