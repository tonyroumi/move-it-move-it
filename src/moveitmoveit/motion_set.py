import torch

from .manifest import COMMAND_DIM, CommandIndex, MotionSpec, load_manifest
from .motion.library import MotionLibrary


class MotionSet:
    """The motions a manifest declares: their clips, plus per-motion command and selection tables.
    Built in one place, so a motion id means the same thing everywhere. Per-env motion state stays
    on the env, and reward and termination tables on ComposedMDP."""

    def __init__(self, specs: list[MotionSpec], device: torch.device | str):
        self.device = device
        self.specs = specs

        self.names = [spec.name for spec in specs]
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        self.num_motions = len(specs)
        self.library = MotionLibrary([spec.file for spec in specs], device)

        self._build_command_tables(specs)
        self._build_sample_prob_table(specs)

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

    def _build_sample_prob_table(self, specs: list[MotionSpec]):
        weights = torch.zeros(self.num_motions, device=self.device)

        for motion_id, spec in enumerate(specs):
            weights[motion_id] = spec.sample_prob

        self.sample_prob = weights / weights.sum()

    def sample_ids(self, num_samples: int) -> torch.Tensor:
        """The one clip-selection distribution, shared by env resets and discriminator batches."""
        return torch.multinomial(self.sample_prob, num_samples=num_samples, replacement=True)

    def sample_commands(self, motion_ids: torch.Tensor, root_positions_xy: torch.Tensor | None = None) -> torch.Tensor:
        """Sample each motion's declared command ranges. A goal command is declared as an offset;
        given the roots' world XY positions (N, 2), motions with a goal get it as the world-space
        position root + offset, which stays put while the robot moves. A zeroed command therefore
        puts the goal where the root already is."""
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
