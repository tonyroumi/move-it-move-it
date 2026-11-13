"""The step state terms read, and the wrapper that composes the terms a run selects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import torch

from isaaclab.utils.math import quat_apply_inverse

from moveitmoveit.mdp.observations import OBSERVATIONS
from moveitmoveit.mdp.rewards import REWARDS
from moveitmoveit.mdp.terminations import TERMINATIONS
from moveitmoveit.motion.manifest import CommandIndex, MotionSpec, RewardSpec, check_known
from moveitmoveit.utils.transforms import world_to_heading_frame_xy

if TYPE_CHECKING:
    from moveitmoveit.motion.reference import TrackingTargets


class BodyState(NamedTuple):
    dof_positions: torch.Tensor
    dof_velocities: torch.Tensor
    root_positions: torch.Tensor
    root_rotations: torch.Tensor
    root_linear_velocities: torch.Tensor
    root_angular_velocities: torch.Tensor
    key_body_positions: torch.Tensor


@dataclass
class StepContext:
    """What reward and termination terms may read at one env step."""

    state: BodyState
    dof_accelerations: torch.Tensor
    commands: torch.Tensor
    actions: torch.Tensor
    previous_actions: torch.Tensor
    sample_targets: Callable[[], TrackingTargets]
    memory: dict[str, torch.Tensor] = field(default_factory=dict)
    standing_pose: torch.Tensor | None = None

    @cached_property
    def targets(self) -> TrackingTargets:
        """The reference pose. Only sampled if a selected term asks for it."""
        return self.sample_targets()

    @property
    def commanded_lin_vel(self) -> torch.Tensor:
        return self.commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1]

    @cached_property
    def local_lin_vel_xy(self) -> torch.Tensor:
        """The root's forward/lateral linear velocity in its body frame, (N, 2)."""
        return quat_apply_inverse(self.state.root_rotations, self.state.root_linear_velocities)[:, :2]

    @cached_property
    def local_yaw_rate(self) -> torch.Tensor:
        """The root's yaw angular velocity in its body frame, (N,)."""
        return quat_apply_inverse(self.state.root_rotations, self.state.root_angular_velocities)[:, 2]

    @property
    def root_positions_xy(self) -> torch.Tensor:
        return self.state.root_positions[:, :2]

    @property
    def commanded_goal(self) -> torch.Tensor:
        """The goal's world-space XY position."""
        return self.commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1]


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


# --- diagnostics ------------------------------------------------------------------------------


def velocity_tracking_metrics(
    name: str, actual: torch.Tensor, commanded: torch.Tensor, tracked: torch.Tensor, unit: str
) -> dict[str, torch.Tensor]:
    """Tracking error and commanded vs. actual velocity, split by whether the command is zero."""
    error = torch.linalg.vector_norm(actual - commanded, dim=-1)
    commanded_speed = torch.linalg.vector_norm(commanded, dim=-1)
    along = torch.sum(actual * commanded, dim=-1) / commanded_speed.clamp_min(1e-12)

    zero = tracked & (commanded_speed == 0)
    nonzero = tracked & (commanded_speed > 0)
    missing = torch.full_like(error, float("nan"))
    return {
        f"{name}_error (zero cmd) [{unit}]": torch.where(zero, error, missing),
        f"{name}_error (nonzero cmd) [{unit}]": torch.where(nonzero, error, missing),
        f"{name}_commanded (nonzero cmd) [{unit}]": torch.where(nonzero, commanded_speed, missing),
        f"{name}_actual_along_cmd (nonzero cmd) [{unit}]": torch.where(nonzero, along, missing),
    }


# --- composition ------------------------------------------------------------------------------


def observation_dim(names: Iterable[str]) -> int:
    """Size of the observation these terms produce."""
    names = list(names)
    check_known(names, OBSERVATIONS, "observations")
    return sum(OBSERVATIONS[name].dim for name in names)


class ComposedMDP:
    """The rewards, terminations and observations one run selected, as the env asks for them."""

    def __init__(
        self,
        specs: list[MotionSpec],
        policy_obs: Iterable[str],
        style_obs: Iterable[str],
        device: torch.device | str,
        joint_names: Sequence[str] | None = None,
    ):
        self._policy_obs = list(policy_obs)
        self._style_obs = list(style_obs)
        check_known(self._policy_obs + self._style_obs, OBSERVATIONS, "observations")

        # Each motion's standing-mode rewards. A motion without a standing block never has an env in
        # standing mode; it keeps its own rewards there anyway, so the standing tables can't change it.
        standing_rewards = [spec.standing.rewards if spec.standing is not None else spec.rewards for spec in specs]

        for spec, standing in zip(specs, standing_rewards):
            check_known(spec.rewards, REWARDS, f"{spec.name}.rewards")
            if "standing_pose" in spec.rewards:
                raise ValueError(f"{spec.name}.rewards: standing_pose is a standing-mode term; declare it in standing.rewards.")
            check_known(standing, REWARDS, f"{spec.name}.standing.rewards")
            check_known(spec.terminations, TERMINATIONS, f"{spec.name}.termination")
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
        self.reward_declared = torch.tensor(
            [[name in spec.rewards for name in self.reward_names] for spec in specs],
            device=device,
            dtype=torch.bool,
        ).reshape(len(specs), len(self.reward_names))
        self.standing_pose_targets = self._standing_pose_table(specs, joint_names, device)

        self._memory: dict[str, torch.Tensor] = {}

    def _standing_pose_table(
        self, specs: list[MotionSpec], joint_names: Sequence[str] | None, device: torch.device | str
    ) -> torch.Tensor | None:
        """Each motion's `standing.pose` as (M, num_joints) in `joint_names` order, 0 where unlisted."""
        if joint_names is None:
            if "standing_pose" in self.reward_names:
                raise ValueError("standing_pose needs the robot's joint_names to place standing.pose targets.")
            return None

        joint_names = list(joint_names)
        targets = torch.zeros(len(specs), len(joint_names), device=device)
        for motion_id, spec in enumerate(specs):
            pose = spec.standing.pose if spec.standing is not None else {}
            check_known(pose, joint_names, f"{spec.name}.standing.pose")
            for joint, value in pose.items():
                targets[motion_id, joint_names.index(joint)] = value
        return targets

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
        if self.standing_pose_targets is not None:
            c.standing_pose = self.standing_pose_targets[motion_ids]

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

    def command_tracking(self, c: StepContext, motion_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-env velocity-command tracking diagnostics for logging, each of shape (N,)."""
        metrics = {}
        if "lin_vel_tracking" in self.reward_names:
            tracked = self.reward_declared[motion_ids, self.reward_names.index("lin_vel_tracking")]
            metrics |= velocity_tracking_metrics("lin_vel", c.local_lin_vel_xy, c.commanded_lin_vel, tracked, "m/s")
        if "yaw_vel_tracking" in self.reward_names:
            tracked = self.reward_declared[motion_ids, self.reward_names.index("yaw_vel_tracking")]
            commanded = c.commands[:, CommandIndex.YAW : CommandIndex.YAW + 1]
            metrics |= velocity_tracking_metrics("yaw_rate", c.local_yaw_rate.unsqueeze(-1), commanded, tracked, "rad/s")
        return metrics

    def reset_memory(self, env_ids: torch.Tensor) -> None:
        """Forget what terms remembered about these envs, e.g. when their episode or goal changes."""
        for value in self._memory.values():
            value[env_ids] = float("nan")

    def died(self, c: StepContext, motion_ids: torch.Tensor) -> torch.Tensor:
        """Envs where a termination their motion enabled has fired."""
        enabled = self.termination_flags[motion_ids]
        if not self.termination_names:
            return enabled.any(dim=-1)
        fired = torch.stack([TERMINATIONS[name](c) for name in self.termination_names], dim=-1)
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
        """The style observation alone, for reference clips, which have no commands."""
        return torch.cat([OBSERVATIONS[name].fn(state, local_frame) for name in self._style_obs], dim=-1)
