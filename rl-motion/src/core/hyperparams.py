"""Hyperparameter dataclasses for RL algorithms."""

from dataclasses import dataclass, field
from typing import Tuple, Any

@dataclass(frozen=True)
class PPOHyperparams:
    clip_epsilon: float = 0.2
    clip_value_loss: bool = True
    value_clip_epsilon: float = 0.2
    learning_rate: float = 3e-4
    num_epochs: int = 10
    num_minibatches: int = 32
    gamma: float = 0.99
    gae_lambda: float = 0.95
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_length: int = 2048
    batch_size: int = 64
    normalize_advantage: bool = True

    @classmethod
    def from_dict(cls, cfg: dict) -> "PPOHyperparams":
        return cls(**{k: v for k, v in cfg.items() if k in cls.__dataclass_fields__})