import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from .humanoid_env_cfg import HumanoidEnvCfg

from .observations.amp_obs import compute_amp_observations
from .observations.proprioception import compute_proprioceptive_obs

from ..commands import COMMAND_DIM
from ..motions.motion_loader import MotionLoader
from ..motions.motion_manager import MotionManager


class HumanoidEnv(DirectRLEnv):
    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_manager = MotionManager(cfg.motion_manifest, self.num_envs, self.device)
        self._motion_loader = MotionLoader(self._motion_manager.motion_files, self.device)

        self._setup_env()

    @property
    def default_state(self) -> tuple[torch.Tensor, ...]:
        default_root_pose = self.robot.data.default_root_pose.torch.clone()
        default_root_vel = self.robot.data.default_root_vel.torch.clone()
        default_root_pose[:, :3] += self.scene.env_origins
        return (
            default_root_pose,
            default_root_vel,
            self.robot.data.default_joint_pos.torch,
            self.robot.data.default_joint_vel.torch,
        )

    @property
    def current_state(self) -> tuple[torch.Tensor, ...]:
        dof_positions = self.robot.data.joint_pos.torch
        dof_velocities = self.robot.data.joint_vel.torch
        root_positions = self.robot.data.body_pos_w.torch[:, self.ref_body_index]
        root_rotations = self.robot.data.body_quat_w.torch[:, self.ref_body_index]
        root_linear_velocities = self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index]
        root_angular_velocities = self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index]
        key_body_positions = self.robot.data.body_pos_w.torch[:, self.key_body_indexes]
        return (
            dof_positions,
            dof_velocities,
            root_positions,
            root_rotations,
            root_linear_velocities,
            root_angular_velocities,
            key_body_positions,
        )

    def _setup_env(self):
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch
        dof_lower_limits = soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index("torso")
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index(["torso"])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # reconfigure AMP observation space according to the number of observations and create the buffer
        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
        )

        self.commands = torch.zeros(
            self.num_envs,
            COMMAND_DIM,
            device=self.device,
        )

        if self.render_enabled:
            self._camera_target_offset = torch.tensor([2.5, 0.0, 0.3], device=self.device)
            self._camera_eye_offset = torch.tensor([4.0, 0.0, -0.4], device=self.device)

    def _setup_scene(self):
        self.robot = self.scene["robot"]

    def _update_camera(self):
        root_pos = self.robot.data.root_pos_w.torch[0]

        target = root_pos + self._camera_target_offset
        eye = target + self._camera_eye_offset

        eye_t = tuple(eye.cpu().tolist())
        target_t = tuple(target.cpu().tolist())

        self.sim.set_camera_view(eye=eye_t, target=target_t)
        try:
            from isaaclab_physx.renderers.kit_viewport_utils import set_kit_renderer_camera_view

            set_kit_renderer_camera_view(eye=eye_t, target=target_t)
        except (ImportError, ModuleNotFoundError):
            pass

    def _write_robot_state(
        self,
        env_ids: torch.Tensor,
        root_state: torch.Tensor,
        dof_positions: torch.Tensor,
        dof_velocities: torch.Tensor,
    ):
        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=dof_positions, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=dof_velocities, env_ids=env_ids)

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target_index(target=target)

        if self.render_enabled:
            self._update_camera()

    def _get_observations(self) -> dict:
        obs = compute_proprioceptive_obs(*self.current_state)
        obs = torch.concatenate((obs, self.commands), dim=-1)

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = compute_proprioceptive_obs(*self.current_state)
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return obs

    def _get_rewards(self) -> torch.Tensor:
        return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)
        # weights = (self._motion_manager.current_reward_weights)
        # rewards = torch.stack(
        #     [
        #         self._motion_tracking_reward(),
        #         self._lin_vel_tracking_reward(),
        #         self._yaw_vel_tracking_reward(),
        #         self._target_hit_reward(),
        #     ],
        #     dim=-1,
        # )

        # return torch.sum(
        #     weights * rewards,
        #     dim=-1,
        # )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = self.robot.data.body_pos_w.torch[:, self.ref_body_index, 2] < 0.65
        # died = self._motion_manager.termination_flags[self._motion_manager.motion_ids]
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        num_samples = env_ids.shape[0]
        motion_id = self._motion_manager.sample_motion(env_ids)
        times = self._motion_loader.sample_times(num_samples, motion_id)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, clip_indexes=motion_id, times=times)
        root_state, dof_pos, dof_vel = self._compute_robot_state(
            env_ids,
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        )

        self._write_robot_state(
            env_ids,
            root_state,
            dof_pos,
            dof_vel,
        )

        self.commands[env_ids] = (
            self._motion_manager.sample_commands(
                env_ids
            )
        )

        amp_observations = self.collect_reference_motions(num_samples, clip_indexes=motion_id, current_times=times)
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

    def _compute_robot_state(
        self,
        env_ids: torch.Tensor,
        dof_positions: torch.Tensor,
        dof_velocities: torch.Tensor,
        body_positions: torch.Tensor,
        body_rotations: torch.Tensor,
        body_linear_velocities: torch.Tensor,
        body_angular_velocities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = torch.cat(
            [
                self.robot.data.default_root_pose.torch[env_ids],
                self.robot.data.default_root_vel.torch[env_ids],
            ],
            dim=-1,
        ).clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]
        return root_state, dof_pos, dof_vel

    def collect_reference_motions(
        self,
        num_samples: int,
        clip_indexes: np.ndarray | None = None,
        current_times: np.ndarray | None = None,
    ) -> torch.Tensor:
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample_history(
            num_samples=num_samples,
            clip_indexes=clip_indexes,
            times=current_times,
            num_steps=self.cfg.num_amp_observations,
        )
        amp_observation = compute_proprioceptive_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)
