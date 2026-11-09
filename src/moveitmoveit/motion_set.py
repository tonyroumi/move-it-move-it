import math

import torch

from .manifest import (
    COMMAND_DIM,
    DEFAULT_NEARBY_GOAL_DISTANCE,
    CommandIndex,
    MotionSpec,
    check_nearby_goal_distance,
    load_manifest,
)
from .motion.library import MotionLibrary


# The directions a nearby goal can be held in, heading frame (+X forward, +Y left): the four arrow
# keys and their normalized diagonals, as play's NearbyGoalKeyboard produces them.
NEARBY_GOAL_DIRECTIONS = torch.tensor(
    [(math.cos(angle), math.sin(angle)) for angle in (i * math.pi / 4 for i in range(8))]
).round(decimals=7)


class MotionSet:
    """The motions a manifest declares: their clips, plus per-motion command and selection tables."""

    def __init__(self, specs: list[MotionSpec], device: torch.device | str):
        self.device = device
        self.specs = specs

        self.names = [spec.name for spec in specs]
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        self.num_motions = len(specs)
        self.library = MotionLibrary([spec.file for spec in specs], device)

        self._build_command_tables(specs)
        self._build_nearby_goal_tables(specs)
        self._build_standing_tables(specs)
        self._build_sample_prob_table(specs)

    @classmethod
    def from_manifest(cls, path: str, device: torch.device | str) -> "MotionSet":
        return cls(load_manifest(path), device)

    def _build_command_tables(self, specs: list[MotionSpec]):
        self.command_low = torch.zeros(self.num_motions, COMMAND_DIM, device=self.device)
        self.command_high = torch.zeros_like(self.command_low)
        self.command_zero_prob = torch.zeros(self.num_motions, device=self.device)
        # motions with a goal, which the commands hold as a world-space goal position: a declared
        # goal command, or a nearby goal the env places relative to the root every step
        self.has_goal = torch.tensor(
            [any(key in spec.commands for key in ("goal_x", "goal_y")) or spec.nearby_goal is not None for spec in specs],
            dtype=torch.bool,
            device=self.device,
        )

        for motion_id, spec in enumerate(specs):
            self.command_zero_prob[motion_id] = spec.zero_command_prob

            for key, (low, high) in spec.commands.items():
                index = CommandIndex[key.upper()]
                self.command_low[motion_id, index] = low
                self.command_high[motion_id, index] = high

    def _build_nearby_goal_tables(self, specs: list[MotionSpec]):
        # motions whose goal is a nearby, heading-relative one (see NearbyGoalSpec)
        self.nearby_goal = torch.tensor(
            [spec.nearby_goal is not None for spec in specs], dtype=torch.bool, device=self.device
        )
        self.nearby_goal_distance = torch.tensor(
            [spec.nearby_goal.distance if spec.nearby_goal else 0.0 for spec in specs], device=self.device
        )
        self.nearby_goal_idle_prob = torch.tensor(
            [spec.nearby_goal.idle_prob if spec.nearby_goal else 0.0 for spec in specs], device=self.device
        )
        self.nearby_goal_switch_prob = torch.tensor(
            [spec.nearby_goal.switch_prob if spec.nearby_goal else 0.0 for spec in specs], device=self.device
        )
        self._nearby_goal_directions = NEARBY_GOAL_DIRECTIONS.to(self.device)

    def use_nearby_goals(self, distance: float | None = None):
        """Make every goal motion's goal a nearby one, for play's nearby_goal joystick mode: at
        `distance` m, or else at the distance each motion trained with (DEFAULT_NEARBY_GOAL_DISTANCE
        for a fixed-goal motion). Raises if a distance doesn't exceed that motion's goal_radius."""
        for motion_id, spec in enumerate(self.specs):
            if not self.has_goal[motion_id]:
                continue
            motion_distance = distance
            if motion_distance is None:
                motion_distance = spec.nearby_goal.distance if spec.nearby_goal else DEFAULT_NEARBY_GOAL_DISTANCE
            goal_radius = spec.standing.goal_radius if spec.standing else 0.0
            check_nearby_goal_distance(motion_distance, goal_radius, f"{spec.name}: nearby goal")
            self.nearby_goal_distance[motion_id] = motion_distance
        self.nearby_goal = self.has_goal.clone()

    def use_world_goals(self):
        """Make every goal a fixed world-space point the caller writes, for play's point_goal mode."""
        self.nearby_goal = torch.zeros_like(self.nearby_goal)

    def sample_goal_directions(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """(N, 2) heading-frame unit directions for nearby-goal motions, uniform over
        NEARBY_GOAL_DIRECTIONS, or none (0) at the motion's idle_prob and for every other motion."""
        choice = torch.randint(len(self._nearby_goal_directions), (motion_ids.shape[0],), device=self.device)
        directions = self._nearby_goal_directions[choice]
        idle = torch.rand(motion_ids.shape[0], device=self.device) < self.nearby_goal_idle_prob[motion_ids]
        return torch.where((self.nearby_goal[motion_ids] & ~idle).unsqueeze(-1), directions, torch.zeros_like(directions))

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

    def sample_ids(self, num_samples: int) -> torch.Tensor:
        """The one clip-selection distribution, shared by env resets and discriminator batches."""
        return torch.multinomial(self.sample_prob, num_samples=num_samples, replacement=True)

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
