from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch

from moveitmoveit.mdp.commands import command_observation
from moveitmoveit.mdp.observations import OBSERVATIONS
from moveitmoveit.mdp.rewards import REWARDS
from moveitmoveit.mdp.terminations import TERMINATIONS
from moveitmoveit.motion.specs import MotionSpec
from moveitmoveit.state import BodyState, StepContext


def observation_dim(names: Iterable[str]) -> int:
    return sum(OBSERVATIONS[name].dim for name in names)


@dataclass
class RewardTable:
    weights: torch.Tensor
    kwargs: dict[str, torch.Tensor]


@dataclass
class TerminationTable:
    enabled: torch.Tensor
    kwargs: dict[str, torch.Tensor]


class MDP:
    """Vectorized rewards, terminations, and observations."""

    def __init__(
        self,
        policy_obs: Iterable[str],
        style_obs: Iterable[str],
        specs: list[MotionSpec],
        device: torch.device | str,
    ):
        self.device = device
        self._policy_obs = list(policy_obs)
        self._style_obs = list(style_obs)
        self._memory: dict[str, torch.Tensor] = {}

        self._rewards = self._build_reward_tables(specs)
        self._terminations = self._build_termination_tables(specs)

    def _build_reward_tables(
        self, specs: list[MotionSpec]
    ) -> dict[str, RewardTable]:
        tables = {}
        for name, term in REWARDS.items():
            configs = [spec.rewards.get(name) for spec in specs]
            if all(config is None for config in configs):
                continue

            weights = torch.tensor(
                [config.weight if config is not None else 0.0 for config in configs],
                device=self.device,
            )

            kwargs = {
                key: torch.tensor(
                    [
                        config.kwargs.get(key, default)
                        if config is not None else default
                        for config in configs
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                for key, default in term.defaults.items()
            }

            tables[name] = RewardTable(weights, kwargs)

        return tables

    def _build_termination_tables(
        self, specs: list[MotionSpec]
    ) -> dict[str, TerminationTable]:
        tables = {}

        for name, term in TERMINATIONS.items():
            enabled = [name in spec.terminations for spec in specs]
            if not any(enabled):
                continue

            kwargs = {
                key: torch.tensor(
                    [
                        spec.terminations.get(name, {}).get(key, default)
                        for spec in specs
                    ],
                    device=self.device,
                    dtype=torch.float32,
                )
                for key, default in term.defaults.items()
            }

            tables[name] = TerminationTable(
                enabled=torch.tensor(enabled, device=self.device),
                kwargs=kwargs,
            )

        return tables

    def reset_memory(self, env_ids: torch.Tensor) -> None:
        for value in self._memory.values():
            value[env_ids] = float("nan")

    def rewards(
        self, c: StepContext, spec_ids: torch.Tensor
    ) -> tuple[torch.Tensor, dict, dict]:
        c.memory = self._memory

        total = torch.zeros(len(spec_ids), device=self.device)
        raw, weighted = {}, {}

        for name, table in self._rewards.items():
            kwargs = {
                key: values[spec_ids]
                for key, values in table.kwargs.items()
            }

            values = REWARDS[name].fn(c, **kwargs)
            contribution = table.weights[spec_ids] * values

            total += contribution
            raw[name] = values.detach()
            weighted[name] = contribution.detach()

        weighted["task"] = total.detach()
        return total, raw, weighted

    def died(
        self, c: StepContext, spec_ids: torch.Tensor
    ) -> torch.Tensor:
        died = torch.zeros(
            len(spec_ids), device=self.device, dtype=torch.bool
        )

        for name, table in self._terminations.items():
            kwargs = {
                key: values[spec_ids]
                for key, values in table.kwargs.items()
            }

            died |= (
                table.enabled[spec_ids]
                & TERMINATIONS[name].fn(c, **kwargs)
            )

        return died

    def observations(
        self,
        context: StepContext,
        local_frame: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.cat(
            [
                *(OBSERVATIONS[name].fn(context.state, local_frame) for name in self._policy_obs),
                command_observation(context, local_frame),
                context.behaviors,
                context.phase.unsqueeze(-1),
            ],
            dim=-1,
        )
        return policy, self.style_obs(context.state, local_frame)

    def style_obs(self, state: BodyState, local_frame: bool) -> torch.Tensor:
        return torch.cat([OBSERVATIONS[name].fn(state, local_frame) for name in self._style_obs], dim=-1)
