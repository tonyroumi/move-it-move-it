import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR


# The agent config each `train.py --algo` loads, by the gym kwarg it is registered under.
ALGO_CFG_ENTRY_POINTS = {
    "AMP": "amp_cfg_entry_point",
    "PPO": "ppo_cfg_entry_point",
}

gym.register(
    id="MoveIt-Humanoid-v0",
    entry_point=f"{__name__}.env:MotionLearningEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:MotionLearningEnvCfg",
        # the default, for scripts that only need the env config
        "agent_cfg_entry_point": f"{CONFIGS_DIR}/agents/amp.yaml",
        "amp_cfg_entry_point": f"{CONFIGS_DIR}/agents/amp.yaml",
        "ppo_cfg_entry_point": f"{CONFIGS_DIR}/agents/ppo.yaml",
    },
)
