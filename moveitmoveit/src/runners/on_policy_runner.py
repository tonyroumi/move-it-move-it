from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from gymnasium.vector import AsyncVectorEnv
import torch
import torch.optim as optim
import gymnasium as gym

from moveitmoveit.src.algo.base import BaseAlgo
from moveitmoveit.src.types import BaseParams

@dataclass(frozen=True)
class OnPolicyRunnerConfig(BaseParams):
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
        runner_config: OnPolicyRunnerConfig = OnPolicyRunnerConfig(),
    ):
        self.env = environment
        self.algo = algorithm
        self.config = runner_config
        self.device = torch.device(runner_config.device)

        obs_dim = environment.observation_space.shape[0]
        action_dim = environment.action_space.shape[0]

        self.algo.init_storage(
            num_envs=self.env.num_envs,
            num_transitions=algorithm.params.num_transitions_per_env,
            obs_dim=obs_dim,
            action_dim=action_dim,
        )

        self.optimizer = optim.Adam(
            self.algo.networks.parameters(),
            lr=algorithm.params.lr,
        )

        self.current_timestep = 0
        self.current_iteration = 0

        os.makedirs(runner_config.checkpoint_dir, exist_ok=True)

    def learn(self) -> None:
        steps_per_iter = self.config.num_envs * self.algo.params.num_transitions_per_env
        total_iterations = self.config.total_timesteps // steps_per_iter

        obs_np, _ = self.env.reset()
        obs = self._to_tensor(obs_np)

        for iteration in range(total_iterations):
            # --- Collect rollout ---
            for _ in range(self.algo.params.num_transitions_per_env):
                with torch.no_grad():
                    actions = self.algo.act(obs.to(self.device))

                next_obs_np, reward, terminated, truncated, info = self.env.step(
                    actions.squeeze(0).cpu().numpy()
                )

                self.algo.process_env_step(
                    rewards=torch.tensor([reward], dtype=torch.float32, device=self.device),
                    dones=torch.tensor([terminated], dtype=torch.float32, device=self.device),
                    infos=info,
                )

                done = terminated | truncated
                if done:
                    self.env.reset()

                obs = self._to_tensor(next_obs_np)

            # --- Compute returns and update ---
            with torch.no_grad():
                last_values = self.algo.get_value(obs)
            self.algo.compute_returns(last_values)

            self.algo.update(self.optimizer)

            self.current_timestep += steps_per_iter
            self.current_iteration += 1

            if self.current_iteration % self.config.log_interval == 0:
                print(
                    f"[iter {self.current_iteration:>6}] "
                    f"timestep: {self.current_timestep:>10,}"
                )

            if self.current_iteration % self.config.checkpoint_interval == 0:
                path = os.path.join(
                    self.config.checkpoint_dir,
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

    def _to_tensor(self, obs_np) -> torch.Tensor:
        return torch.tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
