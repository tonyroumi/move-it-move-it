import torch
import yaml

from moveitmoveit.mdp.commands import COMMAND_DIM, CommandIndex
from moveitmoveit.motion.specs import MotionSpec, parse_motion


def manifest_files(path: str) -> list[str]:
    """The clip file of each manifest motion, in motion id order."""
    with open(path) as f:
        motions = yaml.safe_load(f)["motions"]
    return [motion["file"] for motion in motions.values()]


def load_manifest(path: str, durations: list[float]) -> list[MotionSpec]:
    """Read a manifest. A motion's position in the returned list is its motion id."""
    with open(path) as f:
        motions = yaml.safe_load(f)["motions"]
    return [parse_motion(name, motion, durations[i]) for i, (name, motion) in enumerate(motions.items())]


class Manifest:
    """The motions a manifest declares"""

    def __init__(self, specs: list[MotionSpec], device: torch.device | str):
        self.device = device
        self.specs = specs

        self.names = [spec.name for spec in specs]
        self.name_to_id = {name: i for i, name in enumerate(self.names)}
        self.num_motions = len(specs)
        self.files = [spec.file for spec in specs]

        self._build_command_tables(specs)
        self._build_sample_prob_table(specs)
        self._build_behavior_tables(specs)
        self._build_trigger_rates_and_weights()

        self.loop = torch.tensor([spec.loop for spec in specs], dtype=torch.bool, device=self.device)
        self.random_start = torch.tensor([spec.random_start for spec in specs], dtype=torch.bool, device=self.device)
        self.amp = torch.tensor([spec.amp for spec in specs], dtype=torch.bool, device=self.device)
        self.durations = torch.tensor([spec.duration for spec in specs], dtype=torch.float, device=self.device)

    @classmethod
    def from_manifest(cls, path: str, device: torch.device | str) -> "Manifest":
        return cls(load_manifest(path), device)

    def style_reward_weight(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """The AMP style reward weight of the motion each env runs, its manifest `amp`."""
        return self.amp[motion_ids]

    def _build_command_tables(self, specs: list[MotionSpec]):
        self.command_low = torch.zeros(self.num_motions, COMMAND_DIM, device=self.device)
        self.command_high = torch.zeros_like(self.command_low)
        self.command_zero_prob = torch.zeros(self.num_motions, device=self.device)
        self.has_goal = torch.zeros(self.num_motions, dtype=torch.bool, device=self.device)

        for motion_id, spec in enumerate(specs):
            self.command_zero_prob[motion_id] = spec.zero_command_prob

            for key, (low, high) in spec.commands.items():
                index = CommandIndex[key.upper()]
                if index in (CommandIndex.GOAL_X, CommandIndex.GOAL_Y):
                    self.has_goal[motion_id] = True
                self.command_low[motion_id, index] = low
                self.command_high[motion_id, index] = high

    def _build_trigger_rates_and_weights(self) -> None:
        """Compute behavior sampling weights and trigger frequency."""
        self.trigger_rate = 0.0
        self.trigger_weights = torch.zeros(self.num_behaviors, device=self.device)

        if self.num_behaviors == 0:
            return

        probabilities = self.sample_prob[self.behavior_motion_ids]
        durations = self.durations[self.behavior_motion_ids]

        # Weight triggers inversely by duration to match target time shares.
        rates = probabilities / durations
        self.trigger_weights = rates / rates.sum()

        # Always-on behaviors restart without key presses, so they need no trigger rate.
        if self.always_on:
            return

        # Choose a trigger rate that achieves the desired behavior time share.
        behavior_share = float(probabilities.sum() / self.sample_prob.sum())
        mean_duration = float((self.trigger_weights * durations).sum())

        self.trigger_rate = behavior_share / (
            (1.0 - behavior_share) * mean_duration
        )

    def _build_sample_prob_table(self, specs: list[MotionSpec]):
        self.sample_prob = torch.tensor([spec.sample_prob for spec in specs], dtype=torch.float, device=self.device)

    def _build_behavior_tables(self, specs: list[MotionSpec]) -> None:
        """A motion is a behavior if it contains a trigger"""
        behaviors = sorted(
            (CommandIndex[spec.trigger.upper()], motion_id)
            for motion_id, spec in enumerate(specs)
            if spec.trigger
        )

        self.num_behaviors = len(behaviors)
        # the command slot each behavior's trigger key is written to, in behavior order
        self.behavior_keys = [int(key) for key, _ in behaviors]
        self.behavior_motion_ids = torch.tensor(
            [motion_id for _, motion_id in behaviors],
            dtype=torch.long,
            device=self.device,
        )

        # Map each motion to its behavior index (-1 for locomotion).
        self.motion_behavior = torch.full(
            (self.num_motions,), -1, dtype=torch.long, device=self.device
        )
        self.motion_behavior[self.behavior_motion_ids] = torch.arange(
            self.num_behaviors, device=self.device
        )

        self.always_on = bool((self.motion_behavior >= 0).all())

    def sample_ids(self, num_samples: int) -> torch.Tensor:
        return torch.multinomial(self.sample_prob, num_samples=num_samples, replacement=True)

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

    def sample_behaviors(self, num_samples: int) -> torch.Tensor:
        """Behavior indices drawn by trigger_weights"""
        return torch.multinomial(self.trigger_weights, num_samples=num_samples, replacement=True)
