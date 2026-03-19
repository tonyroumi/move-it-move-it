from __future__ import annotations

from omegaconf import DictConfig
import hydra

from moveitmoveit.src.factory import make_env, make_algo, make_networks, make_runner
from utils import Logger

@hydra.main(version_base=None, config_path="./configs", config_name="amp_humanoid")
def main(cfg: DictConfig):
    logger = Logger()

    env = make_env(cfg.environment, logger)
    obs_dim = env.observation_space.shape[-1]
    action_dim = env.action_space.shape[-1]

    networks = make_networks(cfg.networks,
        in_channels=obs_dim,
        out_channels=action_dim,
        action_space=env.action_space,
        device=cfg.device
    )
    algo = make_algo(cfg.algorithm, networks, logger)

    runner = make_runner(cfg.runner, env, algo)

    runner.learn()


if __name__ == "__main__":
    main()
