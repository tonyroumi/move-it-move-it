"""
AMP environment.
"""

import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR

##
# Register Gym environments.
##

gym.register(
    id="Humanoid-AMP-Walk",
    entry_point=f"{__name__}.humanoid_amp_env:HumanoidAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.humanoid_amp_env_cfg:HumanoidAmpWalkEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/humanoid_amp_walk.yaml",
    },
)

