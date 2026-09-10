import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR


gym.register(
    id="Humanoid-AMP-Locomotion",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid_amp_locomotion.yaml",
    },
)

gym.register(
    id="Humanoid-Joystick-AMP-Locomotion",
    entry_point=f"{__name__}.joystick_humanoid_amp_env:JoystickHumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joystick_humanoid_amp_env_cfg:JoystickHumanoidAmpWalkEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid_amp_locomotion.yaml",
    },
)

gym.register(
    id="Humanoid-PPO-Locomotion",
    entry_point=f"{__name__}.humanoid_env:HumanoidLocomotionEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid_ppo_locomotion.yaml",
    },
)
