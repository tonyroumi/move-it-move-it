from __future__ import annotations

from typing import Callable
import itertools
import os

import torch
import torch.nn as nn

from isaaclab.envs import DirectRLEnv

from moveitmoveit.models import MLP
from moveitmoveit.resources import RunningStandardScaler
from moveitmoveit.storage import CircularBuffer
from moveitmoveit.utils.logger import Logger

from .amp_cfg import AMPCfg
from .conditioning import StyleConditionTracker, discriminator_input
from ..ppo import PPO


class AMP(PPO):
    """Adversarial Motion Priors (AMP) algorithm.

    With a conditional manifest (its motions declare `style_condition`, so the env reports
    `num_style_conditions` > 0) the discriminator also sees a one-hot behavior condition: real
    samples carry their clip's label, generated ones the condition of the command the policy acted
    on, and each real batch is drawn with exactly the condition mix of the generated batch it is
    scored against. Otherwise it is the usual unconditional discriminator.
    """
    cfg: AMPCfg

    def __init__(self, cfg: AMPCfg, logger: Logger):
        super().__init__(cfg=cfg, logger=logger)

    def init(self, env: DirectRLEnv, cfg: dict):
        self.style_batch: Callable = env.unwrapped.reference.style_batch
        self.num_style_conditions: int = getattr(env.unwrapped, "num_style_conditions", 0)

        super().init(env, cfg)

    def _initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        super()._initialize_models(env, model_cfg)

        self.amp_obs_dim = self.style_batch(1).shape[-1]
        self.discriminator = MLP(
            in_channels=self.amp_obs_dim + self.num_style_conditions,
            out_channels=1,
            **model_cfg["discriminator"]
        ).to(env.unwrapped.device)

        self.disc_obs_step_dim = env.unwrapped.amp_observation_space.shape[-1]
        self._disc_obs_preprocessor = RunningStandardScaler(
            size=self.disc_obs_step_dim
        ).to(env.unwrapped.device)

    def _initialize_optimizer(self) -> None:
        super()._initialize_optimizer()
        self.disc_optimizer = torch.optim.Adam(
            itertools.chain(
                self.discriminator.parameters()
            ),
            lr=self.cfg.disc_lr,
        )

    def _initialize_storage(self, env: DirectRLEnv) -> None:
        super()._initialize_storage(env)
        # conditional: each row is the AMP observation followed by its condition, so a sample and
        # its label are written, overwritten and drawn together
        self._amp_observation_buffer = CircularBuffer(
            capacity=self.cfg.discriminator_buffer_capacity,
            sample_shape=(self.amp_obs_dim + (1 if self.num_style_conditions else 0),),
            device=env.unwrapped.device,
        )
        self._style_condition_tracker = (
            StyleConditionTracker(env.unwrapped.num_envs, env.unwrapped.cfg.num_amp_observations, env.unwrapped.device)
            if self.num_style_conditions
            else None
        )

    def process_env_step(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        amp_obs = infos["amp_obs"]
        conditions = None
        if self.num_style_conditions:
            # the condition of the command each env acted on this step; -1 (standing) has none
            conditions = infos["style_condition"]

        with torch.no_grad():
            disc_logits = self.discriminator(
                discriminator_input(
                    self._disc_obs_preprocessor(amp_obs),
                    None if conditions is None else conditions.clamp_min(0),
                    self.num_style_conditions,
                )
            )
            style_reward = -torch.log(
                torch.maximum(1 - 1 / (1 + torch.exp(-disc_logits)), torch.tensor(0.0001, device=rewards.device))
            ).view(rewards.shape)

        if conditions is None:
            self._amp_observation_buffer.append(amp_obs)
        else:
            # only samples whose whole history is policy motion under one condition
            valid = self._style_condition_tracker.update(conditions, (terminated | truncated).view(-1))
            self._amp_observation_buffer.append(
                torch.cat((amp_obs, conditions.unsqueeze(-1).to(amp_obs.dtype)), dim=-1)[valid]
            )

        weighted_style_reward = style_reward * self.cfg.style_reward_lambda
        if conditions is not None:
            # no condition, no style to score against
            weighted_style_reward = torch.where(
                (conditions < 0).view(rewards.shape), torch.zeros_like(weighted_style_reward), weighted_style_reward
            )
        style_reward_off = infos.get("style_reward_off")
        if style_reward_off is not None:
            # flagged envs (e.g. standing) earn no style reward, but (unconditionally) their AMP
            # observations still reach the discriminator's buffer above; without the mask every env earns style
            weighted_style_reward = torch.where(
                style_reward_off.view(rewards.shape), torch.zeros_like(weighted_style_reward), weighted_style_reward
            )
        rewards = self.cfg.goal_reward_lambda * rewards + weighted_style_reward

        infos["reward_terms_raw"]["style"] = style_reward
        infos["reward_terms"]["style"] = weighted_style_reward

        super().process_env_step(next_observations, rewards, terminated, truncated, infos)

    def update(self) -> None:
        super().update()

        if self._update_step % self.cfg.discriminator_update_interval == 0 and len(self._amp_observation_buffer):
            self._update_discriminator()

    def _update_discriminator(self) -> None:
        """Run one round of discriminator gradient updates."""
        for _ in range(self.cfg.disc_num_updates):
            agent_motion = self._amp_observation_buffer.sample(self.cfg.disc_batch_size)
            if self.num_style_conditions:
                agent_motion, conditions = agent_motion[:, :-1], agent_motion[:, -1].long()
                # real samples with the same conditions, so the label alone says nothing about realness
                ref_motion = self.style_batch(self.cfg.disc_batch_size, conditions)
            else:
                conditions = None
                ref_motion = self.style_batch(self.cfg.disc_batch_size)

            ref_motion = self._disc_obs_preprocessor(ref_motion, train=True)
            agent_motion = self._disc_obs_preprocessor(agent_motion, train=True)

            ref_motion.requires_grad_(True)
            agent_logits = self.discriminator(discriminator_input(agent_motion, conditions, self.num_style_conditions))
            ref_logits = self.discriminator(discriminator_input(ref_motion, conditions, self.num_style_conditions))

            discriminator_loss = 0.5 * (
                nn.BCEWithLogitsLoss()(agent_logits, torch.zeros_like(agent_logits))
                + nn.BCEWithLogitsLoss()(ref_logits, torch.ones_like(ref_logits))
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
            "num_style_conditions": self.num_style_conditions,
        }, f"{path}/{filename}")

    def load_checkpoint(self, path: str, device: torch.device) -> None:
        """Load the agent's models from the specified path."""
        checkpoint = torch.load(path, map_location=device)
        # checkpoints from before conditional AMP have no entry and are unconditional
        saved_conditions = checkpoint.get("num_style_conditions", 0)
        if saved_conditions != self.num_style_conditions:
            raise ValueError(
                f"{path}: its discriminator was trained with {saved_conditions} style conditions, but this"
                f" manifest has {self.num_style_conditions} (0 = unconditional). A discriminator can't switch"
                " between conditional and unconditional; train a new run with this manifest."
            )
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.discriminator.load_state_dict(checkpoint["discriminator"])
        self._obs_preprocessor.load_state_dict(checkpoint["obs_preprocessor"])
        self._disc_obs_preprocessor.load_state_dict(checkpoint["disc_obs_preprocessor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.disc_optimizer.load_state_dict(checkpoint["disc_optimizer"])
