import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR


gym.register(
    id="MoveIt-Humanoid-v0",
    entry_point=f"{__name__}.env:MotionLearningEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:MotionLearningEnvCfg",
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/amp.yaml",
    },
)
