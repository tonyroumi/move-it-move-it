from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Optional, Sequence, Tuple

import torch

from .base import BaseBuffer

class CircularObsBuffer(BaseBuffer):
    def __init__(
        self,
        capacity: int,
        num_envs: int,
        obs_dim: int,
        device: torch.device,
    ) -> None:
        super().__init__(capacity=capacity, device=device)
        self.obs_dim = obs_dim
        self.num_envs = num_envs

        self.observations = torch.zeros(capacity, num_envs, obs_dim, device=device)
        self.next_observations = torch.zeros(capacity, num_envs, obs_dim, device=device)

        self._ptr = 0  # next write position (circular)
        self._size = 0  # current number of valid entries

    @property
    def size(self) -> int:
        return self._size

    @property
    def full(self) -> bool:
        return self._size >= self._capacity

    def add(self, obs: torch.Tensor, next_obs: torch.Tensor = None) -> None:
        """Insert one or more transitions into the buffer.
        """

        # Normalise to 2-D
        if obs.ndim == 1:
            obs = obs.unsqueeze(0)
            next_obs = next_obs.unsqueeze(0)

        # Fast path: contiguous block fits without wrapping
        end = self._ptr + self.num_envs
        if end <= self._capacity:
            self.observations[self._ptr:end].copy_(obs)
            self.next_observations[self._ptr:end].copy_(next_obs)
        else:
            # Wraps around – split into two copies
            first = self._capacity - self._ptr
            second = self.num_envs - first

            self.observations[self._ptr:].copy_(obs[:first])
            self.observations[:second].copy_(obs[first:])

            self.next_observations[self._ptr:].copy_(next_obs[:first])
            self.next_observations[:second].copy_(next_obs[first:])

        self._ptr = end % self._capacity
        self._size = min(self._size + self.num_envs, self._capacity)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Uniformly sample a batch of transitions. """
        assert self._size > 0, "Cannot sample from an empty buffer."
        idx = torch.randint(0, self._size, (batch_size,), device=self.device)
        return (
            self.observations[idx],
            self.next_observations[idx],
        )

    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 1,
        sample_size: Optional[int] = None,
    ) -> Generator[
        Tuple[
            torch.Tensor, torch.Tensor,
        ],
        None,
        None,
    ]:
        """Yield mini-batches of replay data.

        Yields
        ------
        obs, next_obs
        """
        assert self._size > 0, "Cannot generate batches from an empty buffer."

        if sample_size is None:
            sample_size = self._size

        usable = (sample_size // num_mini_batches) * num_mini_batches
        mini_batch_size = usable // num_mini_batches

        for _ in range(num_epochs):
            idx = torch.randint(0, self._size, (usable,), device=self.device)

            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                batch_idx = idx[start:end]

                yield (
                    self.observations[batch_idx],
                    self.next_observations[batch_idx],
                )

    def clear(self) -> None:
        self._ptr = 0
        self._size = 0