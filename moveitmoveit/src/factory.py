from functools import partial
from typing import Any, Dict, List, Callable, Tuple

import torch
from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordVideo
import gymnasium as gym

from moveitmoveit.src.algo import PPO, PPOHyperparams, AMP, AMPHyperparams
from moveitmoveit.src.env import AMPEnv
from moveitmoveit.src.models import GaussianActor, BaseMLP, AMPNetworks, PPONetworks, NetworkContainer, EmpiricalNorm
from moveitmoveit.src.runners import OnPolicyRunner, OnPolicyRunnerParams
from utils import Logger

def make_env(cfg, logger: Logger):
    env_type = cfg.type.lower()

    if env_type == "jax":
        raise NotImplementedError("Jax environment has not been implemented yet")

    def env_factory(
        env_id: str, 
        env_idx: int, 
        logger: Logger,
        record_video_path: str = None, 
        record_video_interval: int = 2000,
        env_kwargs: Dict[str, Any] = {}, 
        wrappers: List[Callable] = [],
        ):
        """
        Creates a factory function that initializes and returns a wrapped Gymnasium environment.
        """
        def _init():
            env = gym.make(
                env_id,
                **env_kwargs,
                logger=logger
            )
            for wrapper in wrappers:
                env = wrapper(env)
            if record_video_path is not None and env_idx == 0:
                env = RecordVideo(env, record_video_path, episode_trigger=lambda x: x % record_video_interval == 0)
            return env
        return _init

    env_kwargs = dict(cfg.get("kwargs", {}))
    wrappers = list(cfg.get("wrappers", []))
    num_envs = cfg.get("num_envs", 1)
    seed = cfg.get("seed", 0)

    if env_type == "gym:cpu":
        vector_env_cls = partial(AsyncVectorEnv, context="fork")
        env = vector_env_cls([
            env_factory(
                env_id=cfg.id,
                env_idx=idx,
                logger=logger,
                env_kwargs=env_kwargs,
                wrappers=wrappers,
            )
            for idx in range(num_envs)
        ])
    elif env_type == "gym:debug":
        env = env_factory(
            env_id=cfg.id,
            env_idx=0,
            logger=logger,
            env_kwargs=env_kwargs,
            wrappers=wrappers,
            )()
        env.num_envs = 1
    else:
        raise ValueError(f"Unknown env type '{env_type}'. Available: ['gym:cpu', 'jax']")

    env.reset(seed=seed)
    return env

def make_networks(
    cfg,
    in_channels: int,
    out_channels: int,
    action_space: Tuple,
    device: str
) -> NetworkContainer:
    """Factory that builds a NetworkContainer with normalizers."""
    actor = GaussianActor(in_channels=in_channels, out_channels=out_channels, **cfg.actor).to(device)
    critic = BaseMLP(in_channels=in_channels, out_channels=1, **cfg.critic).to(device)

    obs_norm = None
    if cfg.get("obs_norm") is not None:
        obs_norm = EmpiricalNorm(shape=(in_channels,), device=device, **cfg.obs_norm)

    action_norm = None
    if cfg.get("action_norm") is not None:
        a_mean = torch.tensor(0.5 * (action_space.high + action_space.low), dtype=torch.float32)
        a_std = torch.tensor(0.5 * (action_space.high - action_space.low), dtype=torch.float32)
        action_norm = EmpiricalNorm(
            shape=a_mean.shape,
            device=device,
            init_mean=a_mean,
            init_std=a_std,
            **cfg.action_norm,
        )

    if cfg.get("discriminator") is not None:
        discriminator = BaseMLP(in_channels=10 * in_channels, out_channels=1, **cfg.discriminator).to(device)

        disc_obs_norm = None
        if cfg.get("disc_obs_norm") is not None:
            disc_obs_norm = EmpiricalNorm(shape=(in_channels,), device=device, **cfg.disc_obs_norm)

        return AMPNetworks(
            actor=actor,
            critic=critic,
            discriminator=discriminator,
            obs_norm=obs_norm,
            action_norm=action_norm,
            disc_obs_norm=disc_obs_norm,
        )

    return PPONetworks(actor=actor, critic=critic, obs_norm=obs_norm, action_norm=action_norm)

def make_algo(cfg, networks: NetworkContainer, logger: Logger):
    name = cfg.name.lower()

    if name == "ppo":
        algo_cls, params_cls = PPO, PPOHyperparams
    if name == "amp":
        algo_cls, params_cls = AMP, AMPHyperparams

    params = params_cls.from_dict(cfg)
    return algo_cls(networks=networks, params=params, logger=logger)

def make_runner(cfg, env, algo, logger: Logger):
    params = OnPolicyRunnerParams.from_dict(cfg)
    return OnPolicyRunner(environment=env, algorithm=algo, params=params, logger=logger)