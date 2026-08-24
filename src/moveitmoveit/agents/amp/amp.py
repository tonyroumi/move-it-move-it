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

    def _initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        super()._initialize_models(env, model_cfg)

        assert hasattr(env.unwrapped, "amp_observation_space")

        amp_obs = env.unwrapped.amp_observation_space
        self.discriminator = MLP(
            in_channels=amp_obs.shape[-1],
            out_channels=1,
            **model_cfg["discriminator"]
        ).to(env.unwrapped.device)

        self._disc_obs_preprocessor = RunningStandardScaler(size=amp_obs.shape[-1]).to(env.unwrapped.device)

    def _initialize_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            itertools.chain(
                self.actor.parameters(),
                self.critic.parameters(),
                self.discriminator.parameters()
            ),
            lr=self.cfg.learning_rate,
        )

    def _initialize_storage(
        self,
        env: DirectRLEnv,
        storage_cfg: dict,
    ) -> None:
        super()._initialize_storage(env, storage_cfg)

        self.buf_capacity = storage_cfg.get("capacity", 1_000_000)

        self._ref_motion_buf = None 
        self._motion_buf = None 

    def process_env_step(
        self,
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        super().process_env_step(next_observations, rewards, terminated, truncated, infos)

        if infos is None or "amp_obs" not in infos:
            return

        # RESUME HERE. Need to make sure items check out. don't know about shapes. but it runs.
        # # need ppo working too 

        amp_obs_size = infos["amp_obs"].shape

        # Lazy load motion buffers after first step
        if self._motion_buf is None:
            self._ref_motion_buf = TensorCircularBuffer(
                capacity=self.buf_capacity,
                sample_shape=(amp_obs_size),
                device=rewards.device,
            )

            self._motion_buf = TensorCircularBuffer(
                capacity=self.buf_capacity,
                sample_shape=(amp_obs_size),
                device=rewards.device,
            )

        # Agent motion
        if "amp_obs" in infos:
            self._motion_buf.append(infos["amp_obs"])

        # Reference motion for environments that reset
        if "ref_amp_obs" in infos:
            dones = (terminated | truncated).view(-1)

            if dones.any():
                self._ref_motion_buf.append(
                    infos["ref_amp_obs"][dones]
                )

    def update(self) -> None:
        rewards = self.storage.rewards
        amp_observations = self._motion_buf.get()

        with torch.no_grad():
            disc_logits = self.discriminator(
                self._disc_obs_preprocessor(amp_observations)
            )
            style_reward = -torch.log(
                torch.maximum(1 - 1 / (1 + torch.exp(-disc_logits)), torch.tensor(0.0001, device=rewards.device))
            ).view(rewards.shape)

        combined_rewards = self.cfg.goal_reward_lambda * rewards + self.cfg.style_reward_lambda * style_reward
        rewards.copy_(combined_rewards)

        super().update()

        # NOTE: unreachable until discriminator training is wired up (see _update_discriminator).
        return

        if self.cfg.discriminator_update_interval % self._update_step == 0:
            self._update_discriminator()

        self._motion_buf.clear()
        self._ref_motion_buf.clear()

    def _update_discriminator(self) -> None:
        """Run one round of discriminator gradient updates."""

        for update in self.cfg.disc_num_updates:
            ref_motion = self._ref_motion_buf.sample(self.cfg.disc_batch_size)
            # agent_motion = self._motion_buf.sample(self.cfg.disc_batch_size)

            # ref_motion = self.disc_obs_processor(ref_motion, train=True)
            # agent_motion = self.disc_obs_processor(agent_motion, train=True)

            # # do discriminator work here. 
            # loss = 0

            # self.optimizer.zero_grad()
            # loss.backward()
            # self.optimizer.step()

            # self._diagnostics["Disc Loss"].append(loss)

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
        }, f"{path}/{filename}")

    def load_checkpoint(self, path: str, device: torch.device) -> None:
        """Load the agent's models from the specified path."""
        checkpoint = torch.load(path, map_location=device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self._obs_preprocessor.load_state_dict(checkpoint["obs_preprocessor"])
        self._disc_obs_preprocessor.load_state_dict(checkpoint["disc_obs_preprocessor"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
