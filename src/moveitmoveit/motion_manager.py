import torch
import yaml

from .commands import COMMAND_DIM, CommandIndex
from .rewards import NUM_REWARDS, REWARD_KWARG_DEFAULTS, RewardIndex
from .terminations import NUM_TERMINATIONS, TerminationIndex


class MotionManager:
    def __init__(
        self,
        manifest_path: str,
        device: torch.device | str,
    ):
        self.device = device

        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)["motions"]

        self.motion_names = list(manifest.keys())
        self.motion_files = [motion["file"] for motion in manifest.values()]
        self.num_motions = len(self.motion_names)

        self.motion_name_to_id = {
            name: i
            for i, name in enumerate(self.motion_names)
        }

        # one-hot task id per motion, used to condition the AMP discriminator on which motion
        # is currently active
        self.task_id_table = torch.eye(self.num_motions, device=device)

        self._build_command_tables(manifest)
        self._build_reward_tables(manifest)
        self._build_termination_tables(manifest)
        self._build_sample_prob_table(manifest)

    def _build_command_tables(self, manifest):
        self.command_low = torch.zeros(self.num_motions, COMMAND_DIM, device=self.device)
        self.command_high = torch.zeros_like(self.command_low)
        self.command_zero_prob = torch.zeros(self.num_motions, device=self.device)

        command_keys = {
            "lin_x": CommandIndex.LIN_X,
            "lin_y": CommandIndex.LIN_Y,
            "yaw": CommandIndex.YAW,
            "binary_key_0": CommandIndex.BINARY_KEY_0,
            "binary_key_1": CommandIndex.BINARY_KEY_1,
        }

        for motion_id, motion_name in enumerate(self.motion_names):
            cfg = manifest[motion_name]
            commands = cfg.get("commands", {})
            zero_prob = cfg.get("zero_command_prob", 0.0)

            self.command_zero_prob[motion_id] = zero_prob

            for key, index in command_keys.items():

                value = commands.get(key, 0.0)

                if isinstance(value, list):
                    low, high = value
                else:
                    low = high = value

                self.command_low[motion_id, index] = low
                self.command_high[motion_id, index] = high

    def _build_reward_tables(self, manifest):
        self.reward_weights = torch.zeros(self.num_motions,NUM_REWARDS,device=self.device)

        reward_map = {
            "motion_tracking": RewardIndex.MOTION_TRACKING,
            "lin_vel_tracking": RewardIndex.LIN_VEL_TRACKING,
            "yaw_vel_tracking": RewardIndex.YAW_VEL_TRACKING,
            "target_hit": RewardIndex.TARGET_HIT,
            "line_following": RewardIndex.LINE_FOLLOWING,
            "action_rate_l2": RewardIndex.ACTION_RATE_L2,
            "joint_acc": RewardIndex.JOINT_ACC,
            "joint_vel": RewardIndex.JOINT_VEL,
        }

        # per-motion kwarg tensors for rewards
        self.reward_kwargs = {
            name: {
                kwarg_name: torch.full((self.num_motions,), default, device=self.device)
                for kwarg_name, default in defaults.items()
            }
            for name, defaults in REWARD_KWARG_DEFAULTS.items()
        }

        for motion_id, motion_name in enumerate(self.motion_names):
            rewards = manifest[motion_name].get("rewards", {})

            if not rewards:
                continue

            for name, spec in rewards.items():
                if isinstance(spec, dict):
                    weight = spec.get("weight", 1.0)
                    kwargs = {k: v for k, v in spec.items() if k != "weight"}
                else:
                    weight = spec
                    kwargs = {}

                self.reward_weights[motion_id,reward_map[name]] = weight

                for kwarg_name, value in kwargs.items():
                    self.reward_kwargs[name][kwarg_name][motion_id] = value

    def _build_termination_tables(self, manifest):
        self.termination_flags = torch.zeros(
            self.num_motions,
            NUM_TERMINATIONS,
            device=self.device,
            dtype=torch.bool,
        )

        termination_map = {
            "deviation_from_motion": TerminationIndex.DEVIATION_FROM_MOTION,
            "unhealthy": TerminationIndex.UNHEALTHY,
        }

        for motion_id, motion_name in enumerate(self.motion_names):
            terminations = manifest[motion_name].get("termination", "unhealthy")

            for name, index in termination_map.items():
                self.termination_flags[motion_id, index] = terminations == name

    def _build_sample_prob_table(self, manifest):
        weights = torch.zeros(self.num_motions, device=self.device)

        for motion_id, motion_name in enumerate(self.motion_names):
            weights[motion_id] = manifest[motion_name].get("sample_prob", 1.0)

        self.motion_sample_prob = weights / weights.sum()

    def sample_motion(
        self,
        num_samples: int,
    ) -> torch.Tensor:
        sampled_motions = torch.multinomial(
            self.motion_sample_prob,
            num_samples=num_samples,
            replacement=True,
        )
        return sampled_motions

    def sample_commands(
        self,
        motion_ids: torch.Tensor,
    ) -> torch.Tensor:

        low = self.command_low[motion_ids]
        high = self.command_high[motion_ids]

        commands = low + torch.rand_like(low) * (high - low)

        zero_prob = self.command_zero_prob[motion_ids]
        zero_mask = torch.rand(motion_ids.shape[0], device=self.device) < zero_prob
        commands[zero_mask] = 0.0

        return commands

    def reward_weights_for(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self.reward_weights[motion_ids]

    def reward_kwargs_for(self, reward_name: str, motion_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-env kwargs (e.g. `scale=...`) for the given reward, ready to pass as `**kwargs`."""
        return {
            kwarg_name: values[motion_ids]
            for kwarg_name, values in self.reward_kwargs.get(reward_name, {}).items()
        }

    def termination_flags_for(self, motion_ids: torch.Tensor) -> torch.Tensor:
        return self.termination_flags[motion_ids]

    def task_ids_for(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """One-hot task ids for arbitrary motion ids. Shape is (len(motion_ids), num_motions)."""
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        return self.task_id_table[motion_ids]
