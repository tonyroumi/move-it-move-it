import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR


gym.register(
    id="Humanoid-AMP-Walk",
    entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidWalkEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid_amp_walk.yaml",
    },
)

gym.register(
    id="Humanoid-PPO-Walk",
    entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidWalkEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid_ppo_walk.yaml",
    },
)
