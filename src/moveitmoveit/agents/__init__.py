from moveitmoveit.agents.amp import AMP, AMPCfg
from moveitmoveit.agents.base import BaseAgent, BaseCfg
from moveitmoveit.agents.ppo import PPO, PPOCfg

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
