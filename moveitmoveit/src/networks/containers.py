from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn

from moveitmoveit.src.networks.actor import GaussianActor
from moveitmoveit.src.networks.base import BaseMLP

class NetworkContainer:
    """Base container for algorithm-specific networks."""
    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device


@dataclass
class PPONetworks(NetworkContainer):
    actor: GaussianActor
    critic: BaseMLP

    def parameters(self) -> Iterator[nn.Parameter]:
        return itertools.chain(self.actor.parameters(), self.critic.parameters())

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
        }

    def load_state_dict(self, state: dict) -> None:
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])

    def to(self, device: torch.device | str) -> PPONetworks:
        self.actor = self.actor.to(device)
        self.critic = self.critic.to(device)
        return self


@dataclass
class AMPNetworks(PPONetworks):
    discriminator: BaseMLP

    def state_dict(self) -> dict:
        d = super().state_dict()
        d["discriminator"] = self.discriminator.state_dict()
        return d

    def load_state_dict(self, state: dict) -> None:
        super().load_state_dict(state)
        self.discriminator.load_state_dict(state["discriminator"])

    def to(self, device: torch.device | str) -> AMPNetworks:
        super().to(device)
        self.discriminator = self.discriminator.to(device)
        return self