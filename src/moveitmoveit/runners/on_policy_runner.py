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

        for timestep in tqdm.tqdm(range(self.cfg.timesteps)):
            with torch.no_grad():
                for _ in range(self.cfg.num_transitions_per_env):

                    actions = self.agent.act(observations)

                    observations, rewards, terminated, timeout, infos = self.env.step(actions)

                    self.agent.process_env_step(
                        rewards,
                        terminated,
                        timeout,
                        infos
                    )

                self.agent.update()


                # compute returns?

                #update 


        # for iteration in range(total_iterations):
        #     # collect rollouts
        #     for _ in range(self.params.num_transitions_per_env):
        #         with torch.no_grad():
        #             actions = self.algo.act(obs)

        #         obs, reward, terminated, truncated, info = self.env.step(
        #             actions
        #         )

        #         # gymnasium envs auto reset
        #         self.algo.process_env_step(
        #             rewards=reward,
        #             terminated=terminated,
        #             truncated=truncated,
        #             infos=info,
        #         )
            
        #     # compute returns and update
        #     with torch.no_grad():
        #         last_values = self.algo.get_value(obs)
        #     self.algo.compute_returns(last_values)

        #     self.algo.update(self.optimizer)

        #     self.current_timestep += steps_per_iter
        #     self.current_iteration += 1

        #     if self.current_iteration % self.params.log_interval == 0:
        #         self.logger.pprint(
        #             iteration=self.current_iteration,
        #             wall_time=time.perf_counter() - train_start,
        #             samples=self.current_timestep,
        #         )

        #     if self.current_iteration % self.params.checkpoint_interval == 0:
        #         path = os.path.join(
        #             self.logger.log_dir,
        #             f"checkpoint_{self.current_iteration}.pt",
        #         )
        #         self.save(path)
        #         self.logger.info(f"  Checkpoint saved → {path}")

    def save(self, path: str) -> None:
        pass

    def load(self, path: str) -> None:
        # ckpt = torch.load(path, map_location=self.params.device)
        pass