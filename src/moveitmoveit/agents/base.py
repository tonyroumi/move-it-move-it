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

    def init(self, env: DirectRLEnv, cfg: dict) -> None:
        """Initialize the agent with the environment and configuration."""
        self._step_dt = env.unwrapped.step_dt # policy frequency

        self._initialize_models(env, cfg["models"])
        self._initialize_optimizer()
        self._initialize_storage(env, cfg["storage"])

    @abstractmethod
    def _initialize_models(self, env: DirectRLEnv, model_cfg: dict) -> None:
        pass

    @abstractmethod
    def _initialize_optimizer(self) -> None:
        pass

    @abstractmethod
    def _initialize_storage(self, env: DirectRLEnv, num_transitions_per_env: int, storage_cfg: dict) -> None:
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
        infos = infos or {}
        infos.update({
            "rewards": rewards,
            "dones": terminated | truncated,
            "step_dt": self._step_dt,
        })
        self.logger.add_env_info(infos)
        self.logger.env_step()

    @abstractmethod
    def update(self) -> None:
        """Run gradient updates."""

    @abstractmethod
    def inference(self) -> None:
        """Inference Mode """

    @abstractmethod
    def train(self) -> None:
        """Train mode"""

    @abstractmethod
    def write_checkpoint(self, timestep: int, filename: str | None = None) -> None:
        """Save the agent's models to the specified path."""
        pass

    @abstractmethod
    def load_checkpoint(self, path: str, device: torch.device) -> None:
        """Load the agent's models from the specified path."""
        pass
