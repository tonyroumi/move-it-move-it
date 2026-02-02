from motion.envs.make_env import make_env

from utils import DataUtils

def main():

   env_config = DataUtils.load_yaml("configs/environment/MountainCar.yaml")
   env_kwargs = env_config.get("env_kwargs", {})
   env = make_env(
      env_id = env_config["id"],
      env_type=env_config.get("type"),
      num_envs=env_config.get("num_envs"),
      seed=env_config.get("seed"),
      max_episode_steps=env_config.get("max_episode_steps", 200),
      env_kwargs=env_kwargs
   )

if __name__ == "__main__":
   main()