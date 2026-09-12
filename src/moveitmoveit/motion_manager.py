import torch
import yaml

from .commands import COMMAND_DIM, CommandIndex
from .rewards import NUM_REWARDS, RewardIndex
from .terminations import NUM_TERMINATIONS, TerminationIndex


class MotionManager:
    def __init__(
        self,
        manifest_path: str,
        num_envs: int,
        device: torch.device | str,
    ):
        self.device = device
        self.num_envs = num_envs

        with open(manifest_path, "r") as f:
            manifest = yaml.safe_load(f)["motions"]

        self.motion_names = list(manifest.keys())
        self.motion_files = [motion["file"] for motion in manifest.values()]
        self.num_motions = len(self.motion_names)

        self.motion_name_to_id = {
            name: i
            for i, name in enumerate(self.motion_names)
        }

        self.motion_ids = torch.zeros(
            num_envs,
            dtype=torch.long,
            device=device,
        )

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

        print("[MotionManager] Building command tables:")

        for motion_id, motion_name in enumerate(self.motion_names):
            cfg = manifest[motion_name]
            commands = cfg.get("commands", {})
            zero_prob = cfg.get("zero_command_prob", 0.0)

            self.command_zero_prob[motion_id] = zero_prob

            motion_command_summary = {}

            for key, index in command_keys.items():

                value = commands.get(key, 0.0)

                if isinstance(value, list):
                    low, high = value
                else:
                    low = high = value

                self.command_low[motion_id, index] = low
                self.command_high[motion_id, index] = high

                motion_command_summary[key] = (low, high)

            print(f"  [{motion_id}] '{motion_name}': {motion_command_summary}, zero_command_prob={zero_prob}")

    def _build_reward_tables(self, manifest):
        self.reward_weights = torch.zeros(self.num_motions,NUM_REWARDS,device=self.device)

        reward_map = {
            "motion_tracking": RewardIndex.MOTION_TRACKING,
            "lin_vel_tracking": RewardIndex.LIN_VEL_TRACKING,
            "yaw_vel_tracking": RewardIndex.YAW_VEL_TRACKING,
            "target_hit": RewardIndex.TARGET_HIT,
        }

        print("[MotionManager] Building reward tables:")

        for motion_id, motion_name in enumerate(self.motion_names):
            rewards = manifest[motion_name].get("rewards", {})

            if not rewards:
                print(f"  [{motion_id}] '{motion_name}': no rewards specified (defaults to 0.0 for all)")
                continue

            for name, weight in rewards.items():
                self.reward_weights[motion_id,reward_map[name]] = weight

            print(f"  [{motion_id}] '{motion_name}': {rewards}")

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

        print("[MotionManager] Building termination tables:")

        for motion_id, motion_name in enumerate(self.motion_names):
            terminations = manifest[motion_name].get("termination", "unhealthy")

            if terminations not in termination_map:
                print(f"  [{motion_id}] '{motion_name}': WARNING - unknown termination '{terminations}', no flags set")

            for name, index in termination_map.items():
                self.termination_flags[motion_id, index] = terminations == name

            print(f"  [{motion_id}] '{motion_name}': termination='{terminations}'")

    def _build_sample_prob_table(self, manifest):
        weights = torch.zeros(self.num_motions, device=self.device)

        for motion_id, motion_name in enumerate(self.motion_names):
            weights[motion_id] = manifest[motion_name].get("sample_prob", 1.0)

        self.motion_sample_prob = weights / weights.sum()

        print("[MotionManager] Building motion sampling probabilities:")

        for motion_id, motion_name in enumerate(self.motion_names):
            print(f"  [{motion_id}] '{motion_name}': sample_prob={self.motion_sample_prob[motion_id].item():.4f}")

    def sample_motion(
        self,
        env_ids: torch.Tensor,
    ):
        sampled_motions = torch.multinomial(
            self.motion_sample_prob,
            num_samples=env_ids.shape[0],
            replacement=True,
        ).cpu()
        self.motion_ids[env_ids] = sampled_motions.to(self.device)
        return sampled_motions
    
    def sample_commands(
        self,
        env_ids: torch.Tensor,
    ) -> torch.Tensor:

        motion_ids = self.motion_ids[env_ids]

        low = self.command_low[motion_ids]
        high = self.command_high[motion_ids]

        commands = low + torch.rand_like(low) * (high - low)

        zero_prob = self.command_zero_prob[motion_ids]
        zero_mask = torch.rand(env_ids.shape[0], device=self.device) < zero_prob
        commands[zero_mask] = 0.0

        return commands

    @property
    def current_reward_weights(self):
        return self.reward_weights[
            self.motion_ids
        ]

    @property
    def current_termination_flags(self):
        return self.termination_flags[
            self.motion_ids
        ]

    @property
    def current_task_ids(self) -> torch.Tensor:
        """One-hot task id of each env's currently active motion. Shape is (num_envs, num_motions)."""
        return self.task_id_table[self.motion_ids]

    def task_ids_for(self, motion_ids: torch.Tensor) -> torch.Tensor:
        """One-hot task ids for arbitrary motion ids. Shape is (len(motion_ids), num_motions)."""
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device)
        return self.task_id_table[motion_ids]
