import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from .env_cfg import MotionLearningEnvCfg

from moveitmoveit.utils import transforms

from .commands import COMMAND_DIM, CommandIndex
from .motion_utils.motion_loader import MotionLoader
from .motion_manager import MotionManager
from .rewards import (
    lin_vel_tracking_reward,
    motion_tracking_reward,
    target_hit_reward,
    yaw_vel_tracking_reward,
)
from .terminations import (
    deviation_from_motion_termination,
    unhealthy_termination,
)


class MotionLearningEnv(DirectRLEnv):
    cfg: MotionLearningEnvCfg

    def __init__(self, cfg: MotionLearningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._motion_manager = MotionManager(cfg.motion_manifest, self.num_envs, self.device)
        self._motion_loader = MotionLoader(self._motion_manager.motion_files, self.device)

        self._setup_env()

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

    def current_env_reference_motion(self) -> tuple[torch.Tensor, ...]:
        """Sample the reference motion at each environment's current clip time."""
        current_times = (
            self.motion_start_times
            + self.episode_length_buf.to(torch.float32) * self.step_dt
        )

        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(
            num_samples=self.num_envs,
            clip_indexes=self._motion_manager.motion_ids.cpu().numpy(),
            times=current_times.cpu().numpy(),
        )

        if self.cfg.random_reset:
            body_positions, body_rotations, body_linear_velocities, body_angular_velocities = (
                self._apply_random_yaw_to_reference(
                    self.random_yaw_quat,
                    body_positions,
                    body_rotations,
                    body_linear_velocities,
                    body_angular_velocities,
                )
            )

        return (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        )

    def collect_reference_motions(
        self,
        num_samples: int,
        clip_indexes: np.ndarray | None = None,
        current_times: np.ndarray | None = None,
        env_ids: torch.Tensor | None = None,
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

        if self.cfg.random_reset:
            if env_ids is not None:
                # samples correspond to specific environments (e.g. the AMP observation history
                # seeded on reset): reuse the yaw actually applied to that env's root.
                quat = self.random_yaw_quat[env_ids]
            else:
                # samples are an arbitrary discriminator batch, unrelated to any environment:
                # rotate by an independent random yaw per sample so the discriminator also sees
                # real motion at randomized headings, not just each clip's original heading.
                quat = transforms.random_yaw_orientation(num_samples, self.device)
            quat = quat.repeat_interleave(self.cfg.num_amp_observations, dim=0)
            body_positions, body_rotations, body_linear_velocities, body_angular_velocities = (
                self._apply_random_yaw_to_reference(
                    quat,
                    body_positions,
                    body_rotations,
                    body_linear_velocities,
                    body_angular_velocities,
                )
            )

        amp_observation = compute_proprioceptive_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
            local_frame=self.cfg.random_reset
        )
        return amp_observation.view(-1, self.amp_observation_size)

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

        self.motion_start_times = torch.zeros(self.num_envs, device=self.device)

        # per-env random yaw applied at reset (identity until the env's first reset); reused to
        # rotate later reference-motion samples to match (see `_compute_robot_state`).
        self.random_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.random_yaw_quat[:, -1] = 1.0

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

        if self.render_enabled and self.cfg.camera_type != "none":
            if self.cfg.camera_type == "facing":
                # in front of the robot, looking back at it
                self._camera_target_offset = torch.tensor([2.5, 0.0, 0.3], device=self.device)
                self._camera_eye_offset = torch.tensor([4.0, 0.0, -0.4], device=self.device)
            elif self.cfg.camera_type == "third-person":
                # behind and above the robot, looking forward over it
                self._camera_target_offset = torch.tensor([1.0, 0.0, 0.5], device=self.device)
                self._camera_eye_offset = torch.tensor([-6.0, 0.0, 3.0], device=self.device)

    def _setup_scene(self):
        self.robot = self.scene["robot"]

    def _update_camera(self):
        if self.cfg.camera_type == "none":
            return

        root_pos = self.robot.data.root_pos_w.torch[0]
        root_quat = self.robot.data.body_quat_w.torch[0:1, self.ref_body_index]  # (1, 4)

        target_offset = transforms.quat_apply_yaw(root_quat, self._camera_target_offset.unsqueeze(0)).squeeze(0)
        eye_offset = transforms.quat_apply_yaw(root_quat, self._camera_eye_offset.unsqueeze(0)).squeeze(0)

        target = root_pos + target_offset
        eye = target + eye_offset

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

        self._resample_commands()

    def _get_observations(self) -> dict:
        obs = compute_proprioceptive_obs(*self.current_state, local_frame=self.cfg.random_reset)
        obs = torch.concatenate((obs, self.commands), dim=-1)

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = compute_proprioceptive_obs(
            *self.current_state,
            local_frame=self.cfg.random_reset
        )
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        return obs

    def _get_rewards(self) -> torch.Tensor:
        weights = self._motion_manager.current_reward_weights  # (N, NUM_REWARDS)

        (
            ref_dof_positions,
            _,
            ref_body_positions,
            _,
            _,
            _,
        ) = self.current_env_reference_motion()

        rewards = torch.stack(
            [
                motion_tracking_reward(
                    self.robot.data.joint_pos.torch,
                    ref_dof_positions[:, self.motion_dof_indexes],
                    self.robot.data.body_pos_w.torch[:, self.key_body_indexes],
                    self.robot.data.body_pos_w.torch[:, self.ref_body_index],
                    ref_body_positions[:, self.motion_key_body_indexes],
                    ref_body_positions[:, self.motion_ref_body_index],
                ),
                lin_vel_tracking_reward(
                    self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index],
                    self.robot.data.body_quat_w.torch[:, self.ref_body_index],
                    self.commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1],
                ),
                yaw_vel_tracking_reward(
                    self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index],
                    self.robot.data.body_quat_w.torch[:, self.ref_body_index],
                    self.commands[:, CommandIndex.YAW],
                ),
                target_hit_reward(self.num_envs, self.device),
            ],
            dim=-1,
        )

        return torch.sum(weights * rewards, dim=-1)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        enabled = self._motion_manager.current_termination_flags

        current_times = self.motion_start_times + self.episode_length_buf.to(torch.float32) * self.step_dt
        # only DOF positions are used below, which are joint-local and unaffected by the per-env
        # random yaw applied at reset (see `_compute_robot_state`), so no rotation is needed here.
        ref_dof_positions, *_ = self._motion_loader.sample(
            num_samples=self.num_envs,
            clip_indexes=self._motion_manager.motion_ids.cpu().numpy(),
            times=current_times.cpu().numpy(),
        )

        terminations = torch.stack(
            [
                deviation_from_motion_termination(
                    self.robot.data.joint_pos.torch,
                    ref_dof_positions[:, self.motion_dof_indexes],
                ),
                unhealthy_termination(
                    self.robot.data.body_pos_w.torch[:, self.ref_body_index, 2],
                ),
            ],
            dim=-1,
        )

        died = torch.any(enabled & terminations, dim=-1)
        return died, time_out

    def _resample_commands(self):
        """Resample commands for envs that have gone `command_resample_steps` steps since their
        last command sample (either at reset, in `_reset_idx`, or from a previous call here)."""
        resample_env_ids = (
            (self.episode_length_buf % self.cfg.command_resample_steps == 0).nonzero(as_tuple=False).squeeze(-1)
        )
        if resample_env_ids.numel() > 0:
            self.commands[resample_env_ids] = self._motion_manager.sample_commands(resample_env_ids)

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        num_samples = env_ids.shape[0]
        motion_id = self._motion_manager.sample_motion(env_ids)
        times = self._motion_loader.sample_times(num_samples, motion_id)
        self.motion_start_times[env_ids] = torch.as_tensor(times, dtype=torch.float32, device=self.device)
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

        amp_observations = self.collect_reference_motions(
            num_samples, clip_indexes=motion_id, current_times=times, env_ids=env_ids
        )
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
        if self.cfg.random_reset:
            random_yaw = transforms.random_yaw_orientation(env_ids.shape[0], self.device)
            self.random_yaw_quat[env_ids] = random_yaw
            root_state[:, 3:7] = transforms.quat_mul(random_yaw, root_state[:, 3:7])
            root_state[:, 7:10] = transforms.quat_apply(random_yaw, root_state[:, 7:10])
            root_state[:, 10:13] = transforms.quat_apply(random_yaw, root_state[:, 10:13])
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]
        return root_state, dof_pos, dof_vel

    def _apply_random_yaw_to_reference(
        self,
        quat: torch.Tensor,
        body_positions: torch.Tensor,
        body_rotations: torch.Tensor,
        body_linear_velocities: torch.Tensor,
        body_angular_velocities: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rotate a sampled reference-motion body state about its root position by `quat`, so it
        matches the per-env random yaw orientation applied to the robot's root at reset
        """
        num_bodies = body_positions.shape[1]
        quat = quat.unsqueeze(1).expand(-1, num_bodies, 4)

        root_positions = body_positions[:, self.motion_ref_body_index : self.motion_ref_body_index + 1]
        offsets = body_positions - root_positions

        body_positions = root_positions + transforms.quat_apply(quat, offsets)
        body_rotations = transforms.quat_mul(quat, body_rotations)
        body_linear_velocities = transforms.quat_apply(quat, body_linear_velocities)
        body_angular_velocities = transforms.quat_apply(quat, body_angular_velocities)

        return body_positions, body_rotations, body_linear_velocities, body_angular_velocities


@torch.jit.script
def compute_proprioceptive_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
    local_frame: bool,
) -> torch.Tensor:
    num_envs = root_positions.shape[0]

    key_offsets = key_body_positions - root_positions.unsqueeze(1)

    if local_frame:
        lin_vel = transforms.quat_apply_inverse(root_rotations, root_linear_velocities)
        ang_vel = transforms.quat_apply_inverse(root_rotations, root_angular_velocities)

        num_key_bodies = key_offsets.shape[1]
        key_rotations = root_rotations.unsqueeze(1).expand(num_envs, num_key_bodies, 4)
        key_positions_flat = transforms.quat_apply_inverse(key_rotations, key_offsets).reshape(num_envs, -1)
    else:
        lin_vel = root_linear_velocities
        ang_vel = root_angular_velocities
        key_positions_flat = key_offsets.reshape(num_envs, -1)

    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            transforms.quaternion_to_tangent_and_normal(root_rotations),
            lin_vel,
            ang_vel,
            key_positions_flat,
        ),
        dim=-1,
    )
