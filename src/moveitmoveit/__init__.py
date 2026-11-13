import gymnasium as gym

from moveitmoveit.utils.paths import CONFIGS_DIR


ALGOS = ("PPO", "AMP", "HIL")
"""The trainers `scripts/train.py --algo` selects, by `agent.class_type`."""


def agent_cfg_entry_point(algo: str) -> str:
    """The task's registry key for `algo`'s agent config, `configs/agents/<algo>.yaml`."""
    if algo not in ALGOS:
        raise ValueError(f"Unknown algorithm {algo!r}; expected one of {ALGOS}.")
    return f"{algo.lower()}_cfg_entry_point"


gym.register(
    id="MoveIt-Humanoid-v0",
    entry_point=f"{__name__}.env:MotionLearningEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:MotionLearningEnvCfg",
        **{agent_cfg_entry_point(algo): f"{CONFIGS_DIR}/agents/{algo.lower()}.yaml" for algo in ALGOS},
    },
)
