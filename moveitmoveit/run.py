from __future__ import annotations

from typing import Any
from functools import partial

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
import hydra
from gymnasium.wrappers import TimeLimit
import numpy as np
import gymnasium as gym

from moveitmoveit.src.factory import make_env, make_algo, make_networks, make_runner
from utils import Logger

@hydra.main(version_base=None, config_path="./configs", config_name="amp_humanoid")
def main(cfg: DictConfig):
    logger = Logger()
    # wrappers =  [partial[TimeLimit](TimeLimit, max_episode_steps=500)]
    #TODO amp not working
    #TODO normalizer

    env = make_env(cfg.environment)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    networks = make_networks(cfg.networks, in_channels=obs_dim, out_channels=action_dim).to(cfg.device)
    algo = make_algo(cfg.algorithm, networks, logger)

    runner = make_runner(cfg.runner, env, algo)

    runner.learn()


if __name__ == "__main__":
    main()
