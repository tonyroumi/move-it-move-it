from typing import Optional, Dict, List, Callable, Any
from functools import partial

from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv

import motion.envs.gymansium as gymansium

def make_env(
   env_id: str,
   env_type: str,
   num_envs: Optional[int] = 1,
   seed: Optional[int] = 0,
   env_kwargs: Dict = dict(),
   wrappers: List[Callable] = [],
   **kwargs: Any
):
   if env_type == "jax":
      raise NotImplementedError("Jax environment has not been implemented yet")

   else:
      context = "fork"
      if gymansium.is_gymansium_env(env_id):
         env_factory = gymansium.env_factory
   
   if env_type == "gym:cpu":
      vector_env_cls = partial(AsyncVectorEnv, context=context)

      env = vector_env_cls([
         env_factory(
            env_id=env_id,
            env_idx=idx,
            env_kwargs=env_kwargs,
            wrappers=wrappers,
            **kwargs
         )
         for idx in range(num_envs)
      ])
   
   env.reset(seed=seed)
   return env