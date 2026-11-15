from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import NamedTuple

import torch

from moveitmoveit.mdp.observations import OBSERVATIONS
from moveitmoveit.mdp.rewards import REWARDS
from moveitmoveit.mdp.terminations import TERMINATIONS
from moveitmoveit.motion.manifest import CommandIndex, check_known
from moveitmoveit.motion.specs import MotionSpec, RewardSpec
from moveitmoveit.utils.transforms import world_to_heading_frame_xy


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
    memory: dict[str, torch.Tensor] = field(default_factory=dict)


# --- commands ---------------------------------------------------------------------------------


def policy_commands(
    state: BodyState, commands: torch.Tensor, has_goal: torch.Tensor, local_frame: bool
) -> torch.Tensor:
    """The commands as the policy observes them: the world-space goal becomes an offset from the
    root (heading frame when `local_frame`), and 0 for envs whose motion has no goal."""
    goal = slice(CommandIndex.GOAL_X, CommandIndex.GOAL_Y + 1)
    offset = commands[:, goal] - state.root_positions[:, :2]
    if local_frame:
        offset = world_to_heading_frame_xy(state.root_rotations, offset)

    observed = commands.clone()
    observed[:, goal] = torch.where(has_goal.unsqueeze(-1), offset, torch.zeros_like(offset))
    return observed


# --- composition ------------------------------------------------------------------------------


def observation_dim(names: Iterable[str]) -> int:
    """Size of the observation these terms produce."""
    return sum(OBSERVATIONS[name].dim for name in names)


class ComposedMDP:
    """The rewards, terminations and observations one run selected, as the env asks for them."""

    def __init__(
        self,
        specs: list[MotionSpec],
        policy_obs: Iterable[str],
        style_obs: Iterable[str],
        device: torch.device | str,
    ):
        self._policy_obs = list(policy_obs)
        self._style_obs = list(style_obs)
        check_known(self._policy_obs + self._style_obs, OBSERVATIONS, "observations")

        # Each motion's standing-mode rewards. A motion without a standing block never has an env in
        # standing mode; it keeps its own rewards there anyway, so the standing tables can't change it.
        standing_rewards = [spec.standing.rewards if spec.standing is not None else spec.rewards for spec in specs]

        for spec, standing in zip(specs, standing_rewards):
            check_known(spec.rewards, REWARDS, f"{spec.name}.rewards")
            check_known(standing, REWARDS, f"{spec.name}.standing.rewards")
            check_known(spec.terminations, TERMINATIONS, f"{spec.name}.termination")
            for name, declared in spec.terminations.items():
                check_known(declared, TERMINATIONS[name].defaults, f"{spec.name}.termination.{name}")
            for name, declared in spec.rewards.items():
                check_known(declared.kwargs, REWARDS[name].defaults, f"{spec.name}.rewards.{name}")
            for name, declared in standing.items():
                check_known(declared.kwargs, REWARDS[name].defaults, f"{spec.name}.standing.rewards.{name}")

        # Registry order, so logs and table columns don't depend on the order a manifest lists terms.
        self.reward_names = [
            name for name in REWARDS if any(name in spec.rewards or name in standing for spec, standing in zip(specs, standing_rewards))
        ]
        self.termination_names = [name for name in TERMINATIONS if any(name in spec.terminations for spec in specs)]

        self.reward_weights, self.reward_kwargs = self._reward_tables([spec.rewards for spec in specs], device)
        self.standing_reward_weights, self.standing_reward_kwargs = self._reward_tables(standing_rewards, device)
        self.termination_flags = torch.tensor(
            [[name in spec.terminations for name in self.termination_names] for spec in specs],
            device=device,
            dtype=torch.bool,
        ).reshape(len(specs), len(self.termination_names))
        # Per-motion arguments (M,) of each selected termination; a motion that doesn't enable a term
        # keeps its defaults, which `termination_flags` masks out anyway.
        self.termination_kwargs = {
            name: {
                kwarg: torch.tensor(
                    [spec.terminations.get(name, {}).get(kwarg, default) for spec in specs], device=device
                )
                for kwarg, default in TERMINATIONS[name].defaults.items()
            }
            for name in self.termination_names
        }
        self._memory: dict[str, torch.Tensor] = {}

    def _reward_tables(
        self, declared: list[dict[str, RewardSpec]], device: torch.device | str
    ) -> tuple[torch.Tensor, dict[str, dict[str, torch.Tensor]]]:
        """Per-motion weights (M, T) and per-term kwargs (M,) of the rewards each motion declares."""
        weights = torch.tensor(
            [[rewards[name].weight if name in rewards else 0.0 for name in self.reward_names] for rewards in declared],
            device=device,
        ).reshape(len(declared), len(self.reward_names))
        kwargs = {
            name: {
                kwarg: torch.tensor(
                    [rewards[name].kwargs.get(kwarg, default) if name in rewards else default for rewards in declared],
                    device=device,
                )
                for kwarg, default in REWARDS[name].defaults.items()
            }
            for name in self.reward_names
        }
        return weights, kwargs

    def rewards(
        self, c: StepContext, motion_ids: torch.Tensor, standing: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, dict, dict]:
        """The total reward, then the raw and the weighted per-term values for logging."""
        c.memory = self._memory

        def kwargs(name: str) -> dict[str, torch.Tensor]:
            values = {kwarg: table[motion_ids] for kwarg, table in self.reward_kwargs[name].items()}
            if standing is not None:
                for kwarg, table in self.standing_reward_kwargs[name].items():
                    values[kwarg] = torch.where(standing, table[motion_ids], values[kwarg])
            return values

        raw = {name: REWARDS[name].fn(c, **kwargs(name)) for name in self.reward_names}

        weights = self.reward_weights[motion_ids]
        if standing is not None:
            weights = torch.where(standing.unsqueeze(-1), self.standing_reward_weights[motion_ids], weights)
        weighted = weights * (torch.stack(list(raw.values()), dim=-1) if raw else weights)
        total = weighted.sum(dim=-1)

        terms = {name: weighted[:, i].detach() for i, name in enumerate(raw)}
        terms["task"] = total.detach()
        return total, {name: value.detach() for name, value in raw.items()}, terms

    def reset_memory(self, env_ids: torch.Tensor) -> None:
        """Forget what terms remembered about these envs, e.g. when their episode or goal changes."""
        for value in self._memory.values():
            value[env_ids] = float("nan")

    def died(self, c: StepContext, motion_ids: torch.Tensor) -> torch.Tensor:
        """Envs where a termination their motion enabled has fired."""
        enabled = self.termination_flags[motion_ids]
        if not self.termination_names:
            return enabled.any(dim=-1)
        fired = torch.stack(
            [
                TERMINATIONS[name].fn(c, **{kwarg: table[motion_ids] for kwarg, table in self.termination_kwargs[name].items()})
                for name in self.termination_names
            ],
            dim=-1,
        )
        return torch.any(enabled & fired, dim=-1)

    def observations(
        self, state: BodyState, commands: torch.Tensor, local_frame: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The policy and the style observation. Each selected term runs once. Commands stay the
        trailing block of the policy observation, which the agents' scaler leaves unscaled."""
        values = {
            name: OBSERVATIONS[name].fn(state, local_frame) for name in dict.fromkeys(self._policy_obs + self._style_obs)
        }
        policy = torch.cat([values[name] for name in self._policy_obs] + [commands], dim=-1)
        style = torch.cat([values[name] for name in self._style_obs], dim=-1)
        return policy, style

    def style_obs(self, state: BodyState, local_frame: bool) -> torch.Tensor:
        return torch.cat([OBSERVATIONS[name].fn(state, local_frame) for name in self._style_obs], dim=-1)
