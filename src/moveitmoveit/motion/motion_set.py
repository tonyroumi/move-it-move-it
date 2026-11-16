import torch

from moveitmoveit.mdp.commands import COMMAND_DIM, CommandIndex
from moveitmoveit.motion.manifest import load_manifest
from moveitmoveit.motion.specs import MotionSpec


class Manifest:
    """The motions a manifest declares"""

    def __init__(self, specs: list[MotionSpec], device: torch.device | str):
        self.specs = specs

        self.names = [spec.name for spec in specs]
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        self.num_motions = len(specs)
        self.files = [spec.file for spec in specs]

        self._build_command_tables(specs)
        self._build_standing_tables(specs)
        self._build_sample_prob_table(specs)
        self._build_behavior_tables(specs)

        self.loop = torch.tensor([spec.loop for spec in specs], dtype=torch.bool, device=self.device)
        self.random_start = torch.tensor([spec.random_start for spec in specs], dtype=torch.bool, device=self.device)
        self.amp = torch.tensor([spec.amp for spec in specs], dtype=torch.bool, device=self.device)

        self.device = device

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
        self.sample_prob = torch.tensor([spec.sample_prob for spec in specs], dtype=torch.float, device=self.device)

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
        self.always_on = not bool(is_locomotion.any())
        """Whether every motion is a behavior. With no locomotion motion to start in or return to, a
        reset starts each env in a behavior and a completed behavior starts the next one at once, so
        a behavior's one-hot and `BEHAVIOR_PHASE` are always on and no key press is needed."""
        # stable-posture completion; behaviors that don't declare it never settle
        self.settle_min_phase = torch.full((self.num_motions,), float("inf"), device=self.device)
        self.settle_max_dof_speed = torch.zeros(self.num_motions, device=self.device)
        for motion_id, spec in enumerate(specs):
            if spec.settle is not None:
                self.settle_min_phase[motion_id] = spec.settle.min_phase
                self.settle_max_dof_speed[motion_id] = spec.settle.max_dof_speed

    def sample_ids(self, num_samples: int) -> torch.Tensor:
        """Motion ids drawn directly from manifest weights, including at reset."""
        motion_ids = torch.multinomial(self.sample_prob, num_samples=num_samples, replacement=True)
        return motion_ids, (motion_ids == self.behavior_motion_ids) 

    def specs_for(self, motion_ids: torch.Tensor) -> list[MotionSpec]:
        """The spec each env is running, in `motion_ids` order."""
        return [self.specs[int(motion_id)] for motion_id in motion_ids.tolist()]

    def sample_commands(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """Sample each motion's declared command ranges. Goal slots stay offsets; the env places them."""
        low = self.command_low[motion_ids]
        high = self.command_high[motion_ids]

        commands = low + torch.rand_like(low) * (high - low)

        zero_prob = self.command_zero_prob[motion_ids]
        zero_mask = torch.rand(motion_ids.shape[0], device=self.device) < zero_prob
        commands[zero_mask] = 0.0
        return commands

    def standing(self, motion_ids: torch.Tensor, commands: torch.Tensor, goal_distance: torch.Tensor) -> torch.Tensor:
        """Which envs' commands ask them to stand still, (N,) bool.

        `goal_distance` is the distance from the root to the stored goal, 0 where the motion has none.
        """
        lin_vel = torch.linalg.vector_norm(commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1], dim=-1)
        yaw_vel = commands[:, CommandIndex.YAW].abs()
        return (
            self.can_stand[motion_ids]
            & (lin_vel <= self.standing_lin_vel_threshold[motion_ids])
            & (yaw_vel <= self.standing_yaw_vel_threshold[motion_ids])
            & (goal_distance <= self.standing_goal_radius[motion_ids])
        )
