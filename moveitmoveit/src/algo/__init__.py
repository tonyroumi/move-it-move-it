from .base import BaseAlgo
from .ppo import PPO, PPOHyperparams
from .amp import AMP, AMPHyperparams

_ALGO_REGISTRY = {
    "ppo": (PPO, PPOHyperparams),
    "amp": (AMP, AMPHyperparams),
}