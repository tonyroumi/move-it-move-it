from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import torch
import torch.optim

from isaaclab.envs import DirectRLEnv

from moveitmoveit.utils.logger import Logger


@dataclass(kw_only=True)
class BaseCfg(ABC):
    class_type: str


class BaseAgent(ABC):
    """ Abstract base class for a learning agent. """

    def __init__(
        self,
        cfg: BaseCfg,
        logger: Logger,
    ):
        self.cfg = cfg
        self.logger = logger

        self.grad_step = 0

    @abstractmethod
    def initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        pass

    @abstractmethod
    def initialize_storage(self, env: DirectRLEnv, num_transitions_per_env: int, storage_cfg: dict) -> None:
        pass

    @abstractmethod
    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample actions for the current observations. """
        pass

    @abstractmethod
    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        """Record reward, done flags, and optional step info, then flush the
        current transition into the rollout buffer. """
        pass

    @abstractmethod
    def update(self, optimizer: torch.optim.Optimizer) -> None:
        """Run gradient updates."""
        pass
