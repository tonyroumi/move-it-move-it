from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
import collections

import torch
import torch.optim
import numpy as np

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

        self._diagnostics = collections.defaultdict(list)

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
        next_observations: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        """Record reward, done flags, and optional step info, then flush the
        current transition into the rollout buffer. """
        pass

    @abstractmethod
    def update(self) -> None:
        """Run gradient updates."""
        pass

    @abstractmethod
    def write_checkpoint(self) -> None:
        """Save the agent's models to the specified path."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str, device: torch.device) -> None:
        """Load the agent's models from the specified path."""
        pass
