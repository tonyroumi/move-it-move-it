from dataclasses import dataclass

import torch

from .manifest import COMMAND_DIM, CommandIndex, MotionSpec, load_manifest
from .library import MotionLibrary


@dataclass
class ResetClips:
    """The clips a batch of reset envs initialize, as `MotionSet.sample_reset` draws them."""

    motion_ids: torch.Tensor
    """The motion each env is placed into, a behavior or a locomotion motion. Shape is (M,)."""
    times: torch.Tensor
    """Each env's time in that motion's clip, in seconds. Shape is (M,)."""
    base_motion_ids: torch.Tensor
    """The locomotion motion each env runs, or returns to once the behavior it resets into
    completes. Shape is (M,)."""
    locomotion_start_times: torch.Tensor
    """Each env's time in its locomotion clip at episode time 0, in seconds. Shape is (M,)."""


class MotionSet:
    """The motions a manifest declares: their clips, plus per-motion command and selection tables.
    Motions with a `trigger` are behaviors, numbered by key slot; the rest are locomotion motions."""

    def __init__(self, specs: list[MotionSpec], device: torch.device | str):
        self.device = device
        self.specs = specs

        self.names = [spec.name for spec in specs]
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        self.num_motions = len(specs)
        self.library = MotionLibrary([spec.file for spec in specs], device)

        self._build_command_tables(specs)
        self._build_standing_tables(specs)
        self._build_sample_prob_table(specs)
        self._build_behavior_tables(specs)
        self.loop = torch.tensor([spec.loop for spec in specs], dtype=torch.bool, device=self.device)
        """Motions whose tracking target repeats its clip instead of holding the last frame."""
        self.random_start = torch.tensor([spec.random_start for spec in specs], dtype=torch.bool, device=self.device)
        """Motions whose clip clock starts at a uniform random time when a reset initializes it, not at 0."""

    @classmethod
    def from_manifest(cls, path: str, device: torch.device | str) -> "MotionSet":
        return cls(load_manifest(path), device)

    def _build_command_tables(self, specs: list[MotionSpec]):
        self.command_low = torch.zeros(self.num_motions, COMMAND_DIM, device=self.device)
        self.command_high = torch.zeros_like(self.command_low)
        self.command_zero_prob = torch.zeros(self.num_motions, device=self.device)
        # motions that declare a goal command, which then means a world-space goal position
        self.has_goal = torch.tensor(
            [any(key in spec.commands for key in ("goal_x", "goal_y")) for spec in specs],
            dtype=torch.bool,
            device=self.device,
        )

        for motion_id, spec in enumerate(specs):
            self.command_zero_prob[motion_id] = spec.zero_command_prob

            for key, (low, high) in spec.commands.items():
                index = CommandIndex[key.upper()]
                self.command_low[motion_id, index] = low
                self.command_high[motion_id, index] = high

    def _build_standing_tables(self, specs: list[MotionSpec]):
        # motions that declare a standing mode, and the command thresholds below which it is on
        self.can_stand = torch.tensor([spec.standing is not None for spec in specs], dtype=torch.bool, device=self.device)
        self.standing_lin_vel_threshold = torch.tensor(
            [spec.standing.lin_vel_threshold if spec.standing else 0.0 for spec in specs], device=self.device
        )
        self.standing_yaw_vel_threshold = torch.tensor(
            [spec.standing.yaw_vel_threshold if spec.standing else 0.0 for spec in specs], device=self.device
        )
        self.standing_goal_radius = torch.tensor(
            [spec.standing.goal_radius if spec.standing else 0.0 for spec in specs], device=self.device
        )

    def _build_sample_prob_table(self, specs: list[MotionSpec]):
        weights = torch.zeros(self.num_motions, device=self.device)

        for motion_id, spec in enumerate(specs):
            weights[motion_id] = spec.sample_prob

        self.sample_prob = weights / weights.sum()

    def _build_behavior_tables(self, specs: list[MotionSpec]):
        behaviors = sorted((CommandIndex[spec.trigger.upper()], motion_id) for motion_id, spec in enumerate(specs) if spec.trigger)
        self.num_behaviors = len(behaviors)
        self.behavior_keys = [int(key) for key, _ in behaviors]
        """The key slot of each behavior."""
        self.behavior_motion_ids = torch.tensor([motion_id for _, motion_id in behaviors], dtype=torch.long, device=self.device)
        self.motion_behavior = torch.full((self.num_motions,), -1, dtype=torch.long, device=self.device)
        """Each motion's behavior index, or -1 for a locomotion motion."""
        self.motion_behavior[self.behavior_motion_ids] = torch.arange(self.num_behaviors, device=self.device)

        is_locomotion = self.motion_behavior < 0
        self.locomotion_sample_prob = torch.where(is_locomotion, self.sample_prob, torch.zeros_like(self.sample_prob))
        self.locomotion_sample_prob /= self.locomotion_sample_prob.sum()

        # stable-posture completion; behaviors that don't declare it never settle
        self.settle_min_phase = torch.full((self.num_motions,), float("inf"), device=self.device)
        self.settle_max_dof_speed = torch.zeros(self.num_motions, device=self.device)
        for motion_id, spec in enumerate(specs):
            if spec.settle is not None:
                self.settle_min_phase[motion_id] = spec.settle.min_phase
                self.settle_max_dof_speed[motion_id] = spec.settle.max_dof_speed

        self._build_trigger_rate()

    def _build_trigger_rate(self):
        """How often training presses each behavior's key, chosen so the agent spends about as much of
        its time in each behavior as the reference motion the discriminator sees does.
        """
        if self.num_behaviors == 0:
            self.trigger_rate = 0.0
            self.trigger_weights = torch.zeros(0, device=self.device)
            return
        p = self.sample_prob[self.behavior_motion_ids]
        d = self.library.duration[self.behavior_motion_ids].clamp_min(self.library.dt)
        self.trigger_weights = (p / d) / (p / d).sum()
        """Probability that a training press is for each behavior."""
        behavior_share = float(p.sum())
        mean_duration = float((self.trigger_weights * d).sum())
        self.trigger_rate = behavior_share / ((1.0 - behavior_share) * mean_duration)
        """Training presses per second while in locomotion."""

    def sample_ids(self, num_samples: int) -> torch.Tensor:
        """The one clip-selection distribution, shared by env resets and discriminator batches."""
        return torch.multinomial(self.sample_prob, num_samples=num_samples, replacement=True)

    def sample_locomotion_ids(self, num_samples: int) -> torch.Tensor:
        """The selection distribution restricted to locomotion motions, for an env that resets into a
        behavior and needs a motion to return to."""
        return torch.multinomial(self.locomotion_sample_prob, num_samples=num_samples, replacement=True)

    def start_times(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Where a reset starts each motion's clip clock: a uniform random time within the clip where
        the motion's `random_start`, else 0. Shape is (N,)."""
        times = self.library.sample_times(motion_ids)
        return torch.where(self.random_start[motion_ids], times, torch.zeros_like(times))

    def sample_reset(self, num_samples: int) -> ResetClips:
        motion_ids = self.sample_ids(num_samples)
        times = self.start_times(motion_ids)
        base_motion_ids = motion_ids.clone()
        locomotion_start_times = times.clone()

        in_behavior = self.motion_behavior[motion_ids] >= 0
        num_in_behavior = int(in_behavior.sum())
        if num_in_behavior > 0:
            base_motion_ids[in_behavior] = self.sample_locomotion_ids(num_in_behavior)
            locomotion_start_times[in_behavior] = self.start_times(base_motion_ids[in_behavior])

        return ResetClips(motion_ids, times, base_motion_ids, locomotion_start_times)

    def sample_commands(self, motion_ids: torch.Tensor, root_positions_xy: torch.Tensor | None = None) -> torch.Tensor:
        """Sample each motion's declared command ranges. Goals are offsets from `root_positions_xy`."""
        low = self.command_low[motion_ids]
        high = self.command_high[motion_ids]

        commands = low + torch.rand_like(low) * (high - low)

        zero_prob = self.command_zero_prob[motion_ids]
        zero_mask = torch.rand(motion_ids.shape[0], device=self.device) < zero_prob
        commands[zero_mask] = 0.0

        if root_positions_xy is not None:
            has_goal = self.has_goal[motion_ids]
            commands[has_goal, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1] += root_positions_xy[has_goal]

        return commands

    def standing(self, motion_ids: torch.Tensor, observed_commands: torch.Tensor) -> torch.Tensor:
        """Which envs' commands ask them to stand still, (N,) bool."""
        lin_vel = torch.linalg.vector_norm(observed_commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1], dim=-1)
        yaw_vel = observed_commands[:, CommandIndex.YAW].abs()
        goal_distance = torch.linalg.vector_norm(observed_commands[:, CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1], dim=-1)
        return (
            self.can_stand[motion_ids]
            & (lin_vel <= self.standing_lin_vel_threshold[motion_ids])
            & (yaw_vel <= self.standing_yaw_vel_threshold[motion_ids])
            & (goal_distance <= self.standing_goal_radius[motion_ids])
        )
