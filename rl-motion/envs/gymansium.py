from typing import Any, Dict, List, Callable

import gymnasium as gym
from gymnasium.wrappers import RecordVideo, TimeLimit

def is_gymansium_env(env_id: str):
    """
    Check if the environment is a gymnasium environment
    """
    try:
        import gymnasium as gym
    except ImportError:
        return False
    from gymnasium.envs.registration import registry
  
    return env_id in registry

def env_factory(
   env_id: str,
   env_idx: int,
   max_episode_steps: int = 200,
   env_kwargs: Dict[str, Any] = {},
   wrappers: List[Callable] = []
) -> Callable:

   def _init():
      env = gym.make(
         env_id,
         **env_kwargs
      )
      env = TimeLimit(env, max_episode_steps=max_episode_steps)
      for wrapper in wrappers:
         env = wrapper(env)
      return env 
   return _init
