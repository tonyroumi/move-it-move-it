from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from isaaclab.envs import DirectRLEnv

from moveitmoveit.models import MLP
from moveitmoveit.utils.logger import Logger

from .amp_cfg import AMPCfg
from ..ppo import PPO

class AMP(PPO):
    """Adversarial Motion Priors (AMP) algorithm. """
    cfg: AMPCfg

    def __init__(self, cfg: AMPCfg, logger: Logger):
        super().__init__(cfg=cfg, logger=logger)

    def initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        super().initialize_models(env, model_cfg)

        self.discriminator = MLP(
            in_channels=env.observation_space.shape[-1],
            out_channels=1,
            **model_cfg["discriminator"]
        ).to(env.unwrapped.device)

    def initialize_storage(
        self,
        env: DirectRLEnv,
        num_transitions_per_env: int,
        storage_cfg: dict,
    ) -> None:
        super().initialize_storage(env, num_transitions_per_env)

        self._ref_motion_buf = torch.zeros(
            storage_cfg["capacity"],
            env.unwrapped.num_envs,
            self.cfg.num_disc_obs_steps,
            device=env.unwrapped.device
        )
        self._motion_buf = torch.zeros(
            storage_cfg["capacity"],
            env.unwrapped.num_envs,
            self.cfg.num_disc_obs_steps,
            device=env.unwrapped.device
        )

    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        super().process_env_step(rewards, terminated, truncated, infos)

        # self._ref_motion_buf ...
        # self._motion_buf ... 

    def update(self, optimizer: torch.optim.Optimizer) -> None:
        super().update(optimizer)

    def _update_discriminator(self, optimizer: torch.optim.Optimizer) -> None:
        """Run one round of discriminator gradient updates."""
        pass
