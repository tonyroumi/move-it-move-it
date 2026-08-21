from .amp import AMP, AMPCfg
from .base import BaseAgent, BaseCfg
from .ppo import PPO, PPOCfg

__all__ = [
    "BaseAgent",
    "BaseCfg",
    "PPO",
    "PPOCfg",
    "AMP",
    "AMPCfg",
]


AGENT_REGISTRY: dict[str, dict[str, type]] = {
    "PPO": (PPO, PPOCfg),
    "AMP": (AMP, AMPCfg),
}


def resolve_agent(name: str) -> tuple[type[BaseAgent], type[BaseCfg]]:
    return AGENT_REGISTRY[name]