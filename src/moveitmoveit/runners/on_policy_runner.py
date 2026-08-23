from __future__ import annotations

from dataclasses import dataclass
import os
import time

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

    num_transitions_per_env: int = 32

    log_interval: int = 1000
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

        self._initialize_agent(cfg)

    def _initialize_agent(self, cfg: dict):
        agent_cls, cfg_cls = resolve_agent(cfg["agent"]["class_type"])

        self.agent = agent_cls(
            cfg = cfg_cls(**cfg["agent"]),
            logger = self.logger
        )
        self.agent.initialize_models(self.env, cfg["models"])
        self.agent.initialize_storage(self.env, self.cfg.num_transitions_per_env, cfg["storage"])

    def learn(self) -> None:
        observations, infos = self.env.reset()

        total_iterations = self.cfg.timesteps // self.cfg.num_transitions_per_env
        for iteration in tqdm.tqdm(range(total_iterations), disable=True):
            with torch.no_grad():
                for _ in range(self.cfg.num_transitions_per_env):

                    actions = self.agent.act(observations)

                    observations, rewards, terminated, timeout, infos = self.env.step(actions)

                    self.agent.process_env_step(
                        next_observations=observations,
                        rewards=rewards,
                        terminated=terminated,
                        truncated=timeout,
                        infos=infos
                    )

            self.agent.update()

            if iteration % self.cfg.checkpoint_interval == 0:
                self.agent.write_checkpoint()
