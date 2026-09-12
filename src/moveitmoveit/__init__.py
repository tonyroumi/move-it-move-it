import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR


gym.register(
    id="MoveIt-Humanoid-v0",
    entry_point=f"{__name__}.envs.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.envs.humanoid_env_cfg:LocomotionEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/amp.yaml",
    },
)
