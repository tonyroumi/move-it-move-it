from functools import partial
from typing import Any, Dict, List, Callable

from gymnasium.vector import AsyncVectorEnv
from gymnasium.wrappers import RecordVideo
import gymnasium as gym

from moveitmoveit.src.algo import PPO, PPOHyperparams, AMP, AMPHyperparams
from moveitmoveit.src.env import AMPEnv
from moveitmoveit.src.networks import GaussianActor, BaseMLP, AMPNetworks, PPONetworks, NetworkContainer
from moveitmoveit.src.runners import OnPolicyRunner, OnPolicyRunnerConfig

def make_env(cfg):
    env_type = cfg.type.lower()

    if env_type == "jax":
        raise NotImplementedError("Jax environment has not been implemented yet")

    def env_factory(
        env_id: str, 
        env_idx: int, 
        record_video_path: str = None, 
        record_video_interval: int = 2000,
        env_kwargs: Dict[str, Any] = {}, 
        wrappers: List[Callable] = []):
        """
        Creates a factory function that initializes and returns a wrapped Gymnasium environment.
        """
        def _init():
            env = gym.make(
                env_id,
                **env_kwargs
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
                env_kwargs=env_kwargs,
                wrappers=wrappers,
            )
            for idx in range(num_envs)
        ])
    elif env_type == "gym:debug":
        env = env_factory(
            env_id=cfg.id,
            env_idx=0,
            env_kwargs=env_kwargs,
            wrappers=wrappers,
            )()
        env.num_envs = 1
    else:
        raise ValueError(f"Unknown env type '{env_type}'. Available: ['gym:cpu', 'jax']")

    env.reset(seed=seed)
    return env

def make_networks(cfg, in_channels: int, out_channels: int) -> NetworkContainer:
    """Factory that builds a NetworkContainer. """
    actor = GaussianActor(in_channels=in_channels, out_channels=out_channels, **cfg.actor)
    critic = BaseMLP(in_channels=in_channels, out_channels=1, **cfg.critic)

    if "discriminator" in cfg and cfg.discriminator is not None:
        # 2x in_channels for now until i implement disc obs
        discriminator = BaseMLP(in_channels=2*in_channels, out_channels=1, **cfg.discriminator)
        return AMPNetworks(actor=actor, critic=critic, discriminator=discriminator)

    return PPONetworks(actor=actor, critic=critic)

def make_algo(cfg, networks: NetworkContainer, logger=None):
    name = cfg.name.lower()

    if name == "ppo":
        algo_cls, params_cls = PPO, PPOHyperparams
    if name == "amp":
        algo_cls, params_cls = AMP, AMPHyperparams

    params = params_cls.from_dict(cfg)
    return algo_cls(networks=networks, params=params, logger=logger)

def make_runner(cfg, env, algo):
    runner_config = OnPolicyRunnerConfig.from_dict(cfg)
    return OnPolicyRunner(environment=env, algorithm=algo, runner_config=runner_config)