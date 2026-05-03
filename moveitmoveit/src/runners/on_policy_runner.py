from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from gymnasium.vector import AsyncVectorEnv
import gymnasium as gym
import numpy as np
import torch
import torch.optim as optim

from utils import Logger
from moveitmoveit.src.algo.base import BaseAlgo
from moveitmoveit.src.types import BaseParams

@dataclass(frozen=True)
class OnPolicyRunnerParams(BaseParams):
    num_transitions_per_env: int = 32
    total_timesteps: int = 2_000_000
    num_envs: int = 1
    device: str = "cuda"

    log_interval: int = 1           # iterations between console logs

    checkpoint_interval: int = 100  # iterations between saves
    checkpoint_dir: str = "checkpoints"

class OnPolicyRunner:
    """Generic on-policy training loop.
    """

    def __init__(
        self,
        environment: AsyncVectorEnv,
        algorithm: BaseAlgo,
        params: OnPolicyRunnerParams,
        logger: Logger,
    ):
        self.env = environment
        self.algo = algorithm
        self.params = params
        self.logger = logger
        self.device = torch.device(params.device)

        obs_dim = environment.observation_space.shape[-1]
        action_dim = environment.action_space.shape[-1]

        self.algo.init_storage(
            num_envs=self.env.num_envs,
            num_transitions=params.num_transitions_per_env,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.optimizer = optim.Adam(
            self.algo.networks.parameters(),
            lr=algorithm.params.lr,
        )

        self.current_iteration = 0

        os.makedirs(params.checkpoint_dir, exist_ok=True)

    def learn(self) -> None:
        steps_per_iter = self.env.num_envs * self.params.num_transitions_per_env
        total_iterations = self.params.total_timesteps // steps_per_iter

        obs_np, info_np = self.env.reset()
        obs = torch.from_numpy(obs_np).to(self.device) #TODO I don't want to do this here.
        self.algo.process_reset(info_np)

        for iteration in range(total_iterations):
            # collect rollouts
            for _ in range(self.params.num_transitions_per_env):
                with torch.no_grad():
                    actions = self.algo.act(obs)

                next_obs_np, reward, terminated, truncated, info_np = self.env.step(
                    actions.cpu().numpy()
                )

                dones = terminated | truncated
                self.algo.process_env_step(
                    rewards=torch.from_numpy(reward).to(self.device),
                    dones=torch.tensor(dones).to(self.device),
                    infos=info_np,
                )

                # When I come back make sure that we are processing dones correctly. 
                # TODO: 
                # 1.) Ensure rollout collection is good.
                # 4.) Update steps.
                
                if dones.any():
                    self.algo.process_reset(info_np, env_ids=dones.nonzero()[0])

                obs = torch.from_numpy(next_obs_np).to(self.device)

            # compute returns and update
            with torch.no_grad():
                last_values = self.algo.get_value(obs)
            self.algo.compute_returns(last_values)

            self.algo.update(self.optimizer)

            self.current_timestep += steps_per_iter
            self.current_iteration += 1

            if self.current_iteration % self.params.log_interval == 0:
                print(
                    f"[iter {self.current_iteration:>6}] "
                    f"timestep: {self.current_timestep:>10,}"
                )

            if self.current_iteration % self.params.checkpoint_interval == 0:
                path = os.path.join(
                    self.params.checkpoint_dir,
                    f"checkpoint_{self.current_iteration}.pt",
                )
                self.save(path)
                print(f"  Checkpoint saved → {path}")

    def save(self, path: str) -> None:
        torch.save(
            {
                "networks": self.algo.networks.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "iteration": self.current_iteration,
                "timestep": self.current_timestep,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.algo.networks.load_state_dict(ckpt["networks"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.current_iteration = ckpt["iteration"]
        self.current_timestep = ckpt["timestep"]