import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from .env_cfg import MotionLearningEnvCfg

from moveitmoveit.utils import transforms

from .commands import COMMAND_DIM, CommandIndex
from .motion.motion_loader import MotionLoader
from .motion_manager import MotionManager
from .rewards import (
    action_rate_l2_reward,
    joint_acc_reward,
    joint_vel_reward,
    lin_vel_tracking_reward,
    line_following_reward,
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

        self._motion_manager = MotionManager(cfg.motion_manifest, self.device)
        self._motion_loader = MotionLoader(self._motion_manager.motion_files, self.device)

        self._setup_env()

    @property
    def default_root_state(self) -> torch.Tensor:
        return torch.cat(
            [
                self.robot.data.default_root_pose.torch,
                self.robot.data.default_root_vel.torch,
            ],
            dim=-1,
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
        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index("torso")
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index(["torso"])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        self.motion_start_times = torch.zeros(self.num_envs, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.extras = {}

        # per-env random yaw applied at reset 
        self.random_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.random_yaw_quat[:, -1] = 1.0

        # reconfigure AMP observation space according to the number of observations and create the buffer;
        # each per-step observation is extended with a one-hot task id of the active motion
        self.task_id_dim = self._motion_manager.num_motions
        amp_observation_space = self.cfg.amp_observation_space + self.task_id_dim
        self.amp_observation_size = self.cfg.num_amp_observations * amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, amp_observation_space), device=self.device
        )

        self.command_dim = COMMAND_DIM
        self.commands = torch.zeros(
            self.num_envs,
            self.command_dim,
            device=self.device,
        )

        # current/previous action, tracked for the action-rate smoothness penalty
        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

        if self.render_enabled and self.cfg.camera_type != "none":
            if self.cfg.camera_type == "facing":
                self._camera_target_offset = torch.tensor([2.5, 0.0, 0.3], device=self.device)
                self._camera_eye_offset = torch.tensor([4.0, 0.0, -0.4], device=self.device)
            elif self.cfg.camera_type == "third-person":
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
        self.previous_actions = self.actions
        self.actions = actions.clone()

    def _apply_action(self):
        # actions arrive already scaled to physical joint targets by the agent
        self.robot.set_joint_position_target_index(target=self.actions)

        if self.render_enabled:
            self._update_camera()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        # avoid penalizing a spurious "jump" between the last action of the previous episode
        # and the first action of the new one
        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        num_samples = env_ids.shape[0]
        self.motion_ids[env_ids] = self._motion_manager.sample_motion(num_samples)
        times = self._motion_loader.sample_times(num_samples, self.motion_ids[env_ids])
        self.motion_start_times[env_ids] = times
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, clip_indexes=self.motion_ids[env_ids], times=times)
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

        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        dones[env_ids] = True
        self._resample_commands(dones=dones)

        amp_observations = self.collect_reference_motions(
            num_samples, clip_indexes=self.motion_ids[env_ids], current_times=times, env_ids=env_ids
        )
        self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, self.cfg.num_amp_observations, -1)

    def _get_observations(self) -> dict:
        obs = compute_proprioceptive_obs(*self.current_state, local_frame=self.cfg.random_reset)
        cmd_obs = torch.concatenate((obs, self.commands), dim=-1)

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]

        self.amp_observation_buffer[:, 0] = torch.cat(
            (obs.clone(), self._motion_manager.task_ids_for(self.motion_ids)), dim=-1
        )
        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        self._resample_commands()

        return cmd_obs

    def _get_rewards(self) -> torch.Tensor:
        weights = self._motion_manager.reward_weights_for(self.motion_ids)
        
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
                    **self._motion_manager.reward_kwargs_for("motion_tracking", self.motion_ids),
                ),
                lin_vel_tracking_reward(
                    self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index],
                    self.robot.data.body_quat_w.torch[:, self.ref_body_index],
                    self.commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1],
                    **self._motion_manager.reward_kwargs_for("lin_vel_tracking", self.motion_ids),
                ),
                yaw_vel_tracking_reward(
                    self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index],
                    self.robot.data.body_quat_w.torch[:, self.ref_body_index],
                    self.commands[:, CommandIndex.YAW],
                    **self._motion_manager.reward_kwargs_for("yaw_vel_tracking", self.motion_ids),
                ),
                target_hit_reward(self.num_envs, self.device),
                line_following_reward(
                    self.robot.data.body_quat_w.torch[:, self.ref_body_index],
                    self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index],
                    self.commands[:, CommandIndex.LIN_X : CommandIndex.LIN_Y + 1],
                    **self._motion_manager.reward_kwargs_for("line_following", self.motion_ids),
                ),
                action_rate_l2_reward(
                    self.actions,
                    self.previous_actions,
                    **self._motion_manager.reward_kwargs_for("action_rate_l2", self.motion_ids),
                ),
                joint_acc_reward(
                    self.robot.data.joint_acc.torch,
                    **self._motion_manager.reward_kwargs_for("joint_acc", self.motion_ids),
                ),
                joint_vel_reward(
                    self.robot.data.joint_vel.torch,
                    **self._motion_manager.reward_kwargs_for("joint_vel", self.motion_ids),
                ),
            ],
            dim=-1,
        )

        total_reward = torch.sum(weights * rewards, dim=-1)

        # which clip each env is currently running, and the raw task reward term, for
        # the agent to track episodic/per-clip reward statistics.
        self.extras["motion_ids"] = self.motion_ids.clone()
        self.extras["reward_terms"] = {"task": total_reward}

        return total_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        ref_dof_positions, *_ = self.current_env_reference_motion()
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

        enabled = self._motion_manager.termination_flags_for(self.motion_ids)
        died = torch.any(enabled & terminations, dim=-1)
        return died, time_out

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
            clip_indexes=self.motion_ids,
            times=current_times,
        )

        if self.cfg.random_reset:
            body_positions, body_rotations, body_linear_velocities, body_angular_velocities = (
                transforms.apply_random_yaw(
                    self.random_yaw_quat,
                    self.motion_ref_body_index,
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
        clip_indexes: torch.Tensor | None = None,
        current_times: torch.Tensor | None = None,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if clip_indexes is None:
            clip_indexes = self._motion_manager.sample_motion(num_samples)

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
                random_yaw = self.random_yaw_quat[env_ids]
            else:
                # samples are an arbitrary discriminator batch, unrelated to any environment:
                # rotate by an independent random yaw per sample so the discriminator also sees
                # real motion at randomized headings, not just each clip's original heading.
                random_yaw = transforms.random_yaw_orientation(num_samples, self.device)
            random_yaw = random_yaw.repeat_interleave(self.cfg.num_amp_observations, dim=0)
            body_positions, body_rotations, body_linear_velocities, body_angular_velocities = (
                transforms.apply_random_yaw(
                    random_yaw,
                    self.motion_ref_body_index,
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

        task_ids = self._motion_manager.task_ids_for(clip_indexes).repeat_interleave(
            self.cfg.num_amp_observations, dim=0
        )
        amp_observation = torch.cat((amp_observation, task_ids), dim=-1)

        return amp_observation.view(-1, self.amp_observation_size)

    def _resample_commands(self, dones: torch.Tensor | None = None):
        """Resample commands for envs that have gone `command_resample_steps` steps since their
        last command sample or whose `dones` flag is set."""
        resample = self.episode_length_buf % (self.cfg.command_resample_steps + 1) == 0

        if dones is not None:
            resample = resample | dones

        resample_env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if resample_env_ids.numel() > 0:
            self.commands[resample_env_ids] = self._motion_manager.sample_commands(
                self.motion_ids[resample_env_ids]
            )

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
        root_state = self.default_root_state[env_ids].clone()
        root_state[:, 0:3] = body_positions[:, self.motion_ref_body_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.1  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, self.motion_ref_body_index]
        root_state[:, 7:10] = body_linear_velocities[:, self.motion_ref_body_index]
        root_state[:, 10:13] = body_angular_velocities[:, self.motion_ref_body_index]

        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        # Do not need to rotate dof pos or vel as they are already described root-relative
        if self.cfg.random_reset:
            random_yaw = transforms.random_yaw_orientation(env_ids.shape[0], self.device)
            self.random_yaw_quat[env_ids] = random_yaw
            root_state[:, 0:3] =  transforms.quat_apply(random_yaw, root_state[:, 0:3])
            root_state[:, 3:7] = transforms.quat_mul(random_yaw, root_state[:, 3:7])
            root_state[:, 7:10] = transforms.quat_apply(random_yaw, root_state[:, 7:10])
            root_state[:, 10:13] = transforms.quat_apply(random_yaw, root_state[:, 10:13])
        return root_state, dof_pos, dof_vel


@torch.jit.script
def compute_proprioceptive_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    body_positions: torch.Tensor,
    local_frame: bool,
) -> torch.Tensor:
    num_envs = root_positions.shape[0]
    num_bodies = body_positions.shape[1]

    # Body positions relative to the root.
    body_offsets = body_positions - root_positions.unsqueeze(1)

    if local_frame:
        # Express root velocities in the root's local frame.
        lin_vel = transforms.quat_apply_inverse(
            root_rotations,
            root_linear_velocities,
        )
        ang_vel = transforms.quat_apply_inverse(
            root_rotations,
            root_angular_velocities,
        )

        # Apply the inverse root rotation to every body offset.
        root_rotations_expanded = root_rotations.unsqueeze(1).expand(
            num_envs, num_bodies, 4
        )
        body_positions_local = transforms.quat_apply_inverse(
            root_rotations_expanded,
            body_offsets,
        )

        body_positions_flat = body_positions_local.reshape(num_envs, -1)

    else:
        lin_vel = root_linear_velocities
        ang_vel = root_angular_velocities
        body_positions_flat = body_offsets.reshape(num_envs, -1)

    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],
            transforms.quaternion_to_tangent_and_normal(root_rotations),
            lin_vel,
            ang_vel,
            body_positions_flat,
        ),
        dim=-1,
    )
