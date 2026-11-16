from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from moveitmoveit.mdp.observations import OBSERVATIONS, command_observation
from moveitmoveit.mdp.rewards import REWARDS
from moveitmoveit.mdp.terminations import TERMINATIONS
from moveitmoveit.motion.specs import MotionSpec
from moveitmoveit.motion.library import MotionFrames


class BodyState(NamedTuple):
    dof_positions: torch.Tensor
    dof_velocities: torch.Tensor
    dof_accelerations: torch.Tensor
    root_positions: torch.Tensor
    root_rotations: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor
    key_body_positions: torch.Tensor


@dataclass
class StepContext:
    """What reward and termination terms may read at one env step."""

    state: BodyState
    reference_state: BodyState
    commands: torch.Tensor
    previous_actions: torch.Tensor
    actions: torch.Tensor
    phase: torch.Tensor
    behaviors: torch.Tensor
    goal_active: torch.Tensor
    memory: dict[str, torch.Tensor] = field(default_factory=dict)


def observation_dim(names: Iterable[str]) -> int:
    """Size of the observation these terms produce."""
    return sum(OBSERVATIONS[name].dim for name in names)


class MDP:
    """Rewards, terminations and observations, computed from the specs the caller provides."""

    def __init__(
        self,
        policy_obs: Iterable[str],
        style_obs: Iterable[str],
        device: torch.device | str,
    ):
        self._policy_obs = list(policy_obs)
        self._style_obs = list(style_obs)
        self._memory: dict[str, torch.Tensor] = {}

        self.device = device

    def reset_memory(self, env_ids: torch.Tensor) -> None:
        """Forget what terms remembered about these envs, e.g. when their episode or goal changes."""
        for value in self._memory.values():
            value[env_ids] = float("nan")

    def rewards(self, c: StepContext, specs: list[MotionSpec]) -> tuple[torch.Tensor, dict, dict]:
        """Compute total reward and raw/weighted per-term values for logging."""
        c.memory = self._memory

        active = [spec.rewards for spec in specs]

        total = torch.zeros(len(specs), device=self.device)
        raw, weighted = {}, {}

        for name, term in REWARDS.items():
            if not any(name in rewards for rewards in active):
                continue

            kwargs = {
                key: torch.tensor(
                    [rewards[name].kwargs.get(key, default) if name in rewards else default for rewards in active],
                    device=self.device,
                    dtype=torch.float,
                )
                for key, default in term.defaults.items()
            }

            values = term.fn(c, **kwargs)
            weights = torch.tensor(
                [rewards[name].weight if name in rewards else 0.0 for rewards in active],
                device=self.device,
            )

            contribution = weights * values
            total += contribution

            raw[name] = values.detach()
            weighted[name] = contribution.detach()

        weighted["task"] = total.detach()
        return total, raw, weighted

    def died(self, c: StepContext, specs: list[MotionSpec]) -> torch.Tensor:
        """Return environments where an enabled termination has fired."""
        died = torch.zeros(len(specs), device=self.device, dtype=torch.bool)

        for name, term in TERMINATIONS.items():
            enabled = torch.tensor(
                [name in spec.terminations for spec in specs],
                device=self.device,
            )
            if not enabled.any():
                continue

            kwargs = {
                key: torch.tensor(
                    [spec.terminations.get(name, {}).get(key, default) for spec in specs],
                    device=self.device,
                    dtype=torch.float,
                )
                for key, default in term.defaults.items()
            }

            died |= enabled & term.fn(c, **kwargs)

        return died

    def observations(
        self,
        context: StepContext,
        local_frame: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy and the style observation."""
        state = context.state

        commands = command_observation(state, context.commands, local_frame, context.goal_active)
        policy = torch.cat(
            [
                *(OBSERVATIONS[name].fn(state, local_frame) for name in self._policy_obs),
                commands,
                context.behaviors,
                context.phase.unsqueeze(-1),
            ],
            dim=-1,
        )
        return policy, self.style_obs(state, local_frame)

    def style_obs(self, state: BodyState | MotionFrames, local_frame: bool) -> torch.Tensor:
        """The style observation alone, for reference clips, which have no commands."""
        return torch.cat([OBSERVATIONS[name].fn(state, local_frame) for name in self._style_obs], dim=-1)
