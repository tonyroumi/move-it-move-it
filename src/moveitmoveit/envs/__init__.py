"""
AMP environment.
"""

import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR

gym.register(
    id="MoveitMoveit-Cartpole",
    entry_point=f"{__name__}.cartpole_env:CartpoleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartpole_env_cfg:CartpoleEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/cartpole.yaml",
    },
)

gym.register(
    id="MoveitMoveit-Humanoid-AMP-Walk",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid_amp_walk.yaml",
    },
)

gym.register(
    id="MoveitMoveit-Humanoid",
    entry_point=f"{__name__}.humanoid_env:HumanoidEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_env_cfg:HumanoidEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid.yaml",
    },
)
