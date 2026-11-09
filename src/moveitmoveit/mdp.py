"""The env's MDP as named terms, and the wrapper that composes the ones a run selects."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import TYPE_CHECKING, NamedTuple

import torch

from .manifest import CommandIndex, MotionSpec, RewardSpec, check_known
from .utils import transforms

if TYPE_CHECKING:
    from .motion.reference import TrackingTargets


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
        return transforms.quat_apply_inverse(self.state.root_rotations, self.state.root_linear_velocities)[:, :2]

    @cached_property
    def local_yaw_rate(self) -> torch.Tensor:
        """The root's yaw angular velocity in its body frame, (N,)."""
        return transforms.quat_apply_inverse(self.state.root_rotations, self.state.root_angular_velocities)[:, 2]

    @property
    def root_positions_xy(self) -> torch.Tensor:
        return self.state.root_positions[:, :2]

    @property
    def commanded_goal(self) -> torch.Tensor:
        """The goal's world-space XY position."""
        return self.commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1]


class ObservationTerm(NamedTuple):
    fn: Callable[[BodyState, bool], torch.Tensor]
    dim: int


class RewardTerm(NamedTuple):
    fn: Callable[..., torch.Tensor]
    defaults: dict[str, float]


OBSERVATIONS: dict[str, ObservationTerm] = {}
REWARDS: dict[str, RewardTerm] = {}
TERMINATIONS: dict[str, Callable[[StepContext], torch.Tensor]] = {}


def observation(dim: int):
    def register(fn):
        OBSERVATIONS[fn.__name__] = ObservationTerm(fn, dim)
        return fn

    return register


def reward(**defaults: float):
    def register(fn):
        REWARDS[fn.__name__] = RewardTerm(fn, defaults)
        return fn

    return register


def termination(fn):
    TERMINATIONS[fn.__name__] = fn
    return fn



@observation(dim=28)
def dof_pos(state: BodyState, local_frame: bool) -> torch.Tensor:
    return state.dof_positions


@observation(dim=28)
def dof_vel(state: BodyState, local_frame: bool) -> torch.Tensor:
    return state.dof_velocities


@observation(dim=1)
def root_height(state: BodyState, local_frame: bool) -> torch.Tensor:
    return state.root_positions[:, 2:3]


@observation(dim=6)
def root_orientation(state: BodyState, local_frame: bool) -> torch.Tensor:
    return transforms.quaternion_to_tangent_and_normal(state.root_rotations)


@observation(dim=3)
def root_lin_vel(state: BodyState, local_frame: bool) -> torch.Tensor:
    if local_frame:
        return transforms.quat_apply_inverse(state.root_rotations, state.root_linear_velocities)
    return state.root_linear_velocities


@observation(dim=3)
def root_ang_vel(state: BodyState, local_frame: bool) -> torch.Tensor:
    if local_frame:
        return transforms.quat_apply_inverse(state.root_rotations, state.root_angular_velocities)
    return state.root_angular_velocities


@observation(dim=12)
def key_body_offsets(state: BodyState, local_frame: bool) -> torch.Tensor:
    offsets = state.key_body_positions - state.root_positions.unsqueeze(1)
    if local_frame:
        rotations = state.root_rotations.unsqueeze(1).expand(-1, offsets.shape[1], -1)
        offsets = transforms.quat_apply_inverse(rotations, offsets)
    return offsets.flatten(1)


# --- rewards ----------------------------------------------------------------------------------


@reward(dof_scale=2.0, key_body_scale=5.0)
def motion_tracking(c: StepContext, dof_scale: torch.Tensor, key_body_scale: torch.Tensor) -> torch.Tensor:
    """DOF positions and root-relative key-body offsets vs. the reference motion."""
    dof_error = torch.sum((c.state.dof_positions - c.targets.dof_positions) ** 2, dim=-1)
    offsets = c.state.key_body_positions - c.state.root_positions.unsqueeze(1)
    ref_offsets = c.targets.key_body_positions - c.targets.root_positions.unsqueeze(1)
    key_body_error = torch.sum((offsets - ref_offsets).flatten(1) ** 2, dim=-1)
    return torch.exp(-dof_scale * dof_error - key_body_scale * key_body_error)


@reward(scale=2.0)
def lin_vel_tracking(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Matching the commanded body-frame forward/lateral linear velocity."""
    return torch.exp(-scale * torch.sum((c.local_lin_vel_xy - c.commanded_lin_vel) ** 2, dim=-1))


@reward(scale=2.0)
def yaw_vel_tracking(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Matching the commanded body-frame yaw angular velocity."""
    return torch.exp(-scale * (c.local_yaw_rate - c.commands[:, CommandIndex.YAW]) ** 2)


@reward(scale=0.5)
def target_hit(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Closing the 2D distance to the commanded point-goal position."""
    return torch.exp(-scale * torch.sum((c.root_positions_xy - c.commanded_goal) ** 2, dim=-1))


@reward(scale=2.0)
def line_following(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Moving along the world-frame direction implied by the commanded body-frame velocity."""
    command = torch.cat((c.commanded_lin_vel, torch.zeros_like(c.commanded_lin_vel[:, :1])), dim=-1)
    commanded_xy = transforms.quat_apply(c.state.root_rotations, command)[:, :2]
    actual_xy = c.state.root_linear_velocities[:, :2]

    commanded_norm = torch.norm(commanded_xy, dim=-1).clamp_min(1e-6)
    actual_norm = torch.norm(actual_xy, dim=-1).clamp_min(1e-6)
    alignment = torch.sum(commanded_xy * actual_xy, dim=-1) / (commanded_norm * actual_norm)
    return torch.exp(-scale * (1.0 - alignment))


@reward(scale=1.0)
def action_rate_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Large frame-to-frame changes in action."""
    return torch.exp(-scale * torch.sum((c.actions - c.previous_actions) ** 2, dim=-1))


@reward(scale=1.0e-4)
def joint_acc(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Large joint accelerations."""
    return torch.exp(-scale * torch.sum(c.dof_accelerations**2, dim=-1))


@reward(scale=1.0e-2)
def joint_vel(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Large joint velocities."""
    return torch.exp(-scale * torch.sum(c.state.dof_velocities**2, dim=-1))


@reward(scale=1.0)
def progress(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """How much closer to the commanded goal position the last step made us. 0 on an env's first
    step and on the first step after its episode or goal changes, so a new goal earns nothing."""
    distance = torch.norm(c.commanded_goal - c.root_positions_xy, dim=-1)
    previous = c.memory.get("goal_distance", distance)
    previous = torch.where(torch.isnan(previous), distance, previous)
    c.memory["goal_distance"] = distance
    return scale * (previous - distance)


# Standing terms. Bounded in [0, 1] and 1 for a still, upright robot over its feet; meant for a
# motion's `standing.rewards`, together with e.g. joint_vel and action_rate_l2.


@reward(scale=4.0)
def root_lin_vel_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Root linear speed, all three axes."""
    return torch.exp(-scale * torch.sum(c.state.root_linear_velocities**2, dim=-1))


@reward(scale=2.0)
def root_ang_vel_l2(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Root angular speed, all three axes."""
    return torch.exp(-scale * torch.sum(c.state.root_angular_velocities**2, dim=-1))


@reward(scale=10.0)
def upright(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """Root tilt, as 1 minus the world-Z component of its up axis (0 upright, 1 lying flat)."""
    up_axis = torch.zeros_like(c.state.root_positions)
    up_axis[:, 2] = 1.0
    tilt = 1.0 - transforms.quat_apply(c.state.root_rotations, up_axis)[:, 2]
    return torch.exp(-scale * tilt)


@reward(scale=20.0)
def support_centered(c: StepContext, scale: torch.Tensor) -> torch.Tensor:
    """XY distance from the root to the midpoint of the feet, the last two key bodies in the env's
    order (right_hand, left_hand, right_foot, left_foot): the root held over its support."""
    feet_midpoint = c.state.key_body_positions[:, -2:, :2].mean(dim=1)
    return torch.exp(-scale * torch.sum((c.root_positions_xy - feet_midpoint) ** 2, dim=-1))


@reward(scale=2.0, tolerance=0.05)
def standing_pose(c: StepContext, scale: torch.Tensor, tolerance: torch.Tensor) -> torch.Tensor:
    """Joint positions vs. the motion's `standing.pose` target. Each joint may deviate up to
    `tolerance` rad for free, room to balance; the squared excess beyond it is summed over joints.
    Only allowed in `standing.rewards`."""
    excess = ((c.state.dof_positions - c.standing_pose).abs() - tolerance.unsqueeze(-1)).clamp_min(0.0)
    return torch.exp(-scale * torch.sum(excess**2, dim=-1))


# --- terminations -----------------------------------------------------------------------------


@termination
def deviation_from_motion(c: StepContext, threshold: float = 1.5) -> torch.Tensor:
    """The tracked pose deviates too far (summed squared DOF error) from the reference motion."""
    return torch.sum((c.state.dof_positions - c.targets.dof_positions) ** 2, dim=-1) > threshold


@termination
def unhealthy(c: StepContext, height_threshold: float = 0.65) -> torch.Tensor:
    """The root body falls below a minimum height."""
    return c.state.root_positions[:, 2] < height_threshold


# --- commands ---------------------------------------------------------------------------------


def policy_commands(
    state: BodyState, commands: torch.Tensor, has_goal: torch.Tensor, local_frame: bool
) -> torch.Tensor:
    """The commands as the policy observes them.

    The goal command is a world-space XY position, which a policy can't use as is, so it becomes
    the goal's offset from the root: in the root's heading frame when `local_frame`, else along the
    world axes. Envs whose motion has no goal (`has_goal`, shape (N,)) observe 0 there.
    """
    goal = slice(CommandIndex.GOAL_X, CommandIndex.GOAL_Y + 1)
    offset = commands[:, goal] - state.root_positions[:, :2]
    if local_frame:
        offset = transforms.world_to_heading_frame_xy(state.root_rotations, offset)

    observed = commands.clone()
    observed[:, goal] = torch.where(has_goal.unsqueeze(-1), offset, torch.zeros_like(offset))
    return observed


# --- diagnostics ------------------------------------------------------------------------------


def velocity_tracking_metrics(
    name: str, actual: torch.Tensor, commanded: torch.Tensor, tracked: torch.Tensor, unit: str
) -> dict[str, torch.Tensor]:
    """Tracking error and commanded vs. actual velocity, split by whether the command is zero.
    """
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
    """The rewards, terminations and observations one run selected, as the env asks for them.
    """

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
