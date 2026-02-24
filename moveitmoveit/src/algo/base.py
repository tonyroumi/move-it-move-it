from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.optim

from moveitmoveit.src.networks.containers import NetworkContainer
from moveitmoveit.src.types import BaseParams

from utils import Logger

class BaseAlgo(ABC):
    """ Abstract base class for on-policy algorithms. """

    def __init__(
        self,
        networks: NetworkContainer,
        params: BaseParams,
        logger: Logger = None,
    ):
        self.networks = networks
        self.params = params
        self.logger = logger if logger is not None else Logger()

    @abstractmethod
    def init_storage(
        self,
        num_envs: int,
        num_transitions: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        ...

    @abstractmethod
    def act(self, observations: torch.Tensor) -> torch.Tensor:
        """Sample actions for the current observations. """
        ...

    @abstractmethod
    def process_env_step(
        self,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        """Record reward, done flags, and optional step info, then flush the
        current transition into the rollout buffer. """
        ...

    @abstractmethod
    def compute_returns(self, last_values: torch.Tensor) -> None:
        """Estimate returns and advantages over the collected rollout."""
        ...

    @abstractmethod
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute the value estimate for the given observations. """
        ...

    @abstractmethod
    def update(self, optimizer: torch.optim.Optimizer) -> None:
        """Run one epoch of gradient updates using the stored rollout data."""
        ...