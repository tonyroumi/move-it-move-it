import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from .env_cfg import MotionLearningEnvCfg

from moveitmoveit.utils import transforms

from .manifest import COMMAND_DIM
from .mdp import BodyState, ComposedMDP, StepContext, observation_dim, policy_commands
from .motion.reference import ReferenceMotions, ResetFrames, SkeletonMap, TrackingTargets
from .motion_set import MotionSet


class MotionLearningEnv(DirectRLEnv):
    cfg: MotionLearningEnvCfg

    def __init__(self, cfg: MotionLearningEnvCfg, render_mode: str | None = None, **kwargs):
        # DirectRLEnv builds its gym spaces from these during __init__
        cfg.observation_space = observation_dim(cfg.policy_obs) + COMMAND_DIM
        cfg.amp_observation_space = observation_dim(cfg.style_obs)
        super().__init__(cfg, render_mode, **kwargs)

        self.motions = MotionSet.from_manifest(cfg.motion_manifest, self.device)
        self.mdp = ComposedMDP(self.motions.specs, cfg.policy_obs, cfg.style_obs, self.device)

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
    def current_state(self) -> BodyState:
        return BodyState(
            self.robot.data.joint_pos.torch,
            self.robot.data.joint_vel.torch,
            self.robot.data.body_pos_w.torch[:, self.ref_body_index],
            self.robot.data.body_quat_w.torch[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index],
            self.robot.data.body_pos_w.torch[:, self.key_body_indexes],
        )

    def _setup_env(self):
        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index("torso")
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.reference = ReferenceMotions(
            self.motions.sample_ids,
            self.motions.library,
            SkeletonMap(
                self.motions.library.get_dof_index(self.robot.data.joint_names),
                self.motions.library.get_body_index(["torso"])[0],
                self.motions.library.get_body_index(key_body_names),
            ),
            self.cfg.num_amp_observations,
            self.cfg.random_reset,
            self.mdp.style_obs,
        )

        self.motion_start_times = torch.zeros(self.num_envs, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.extras = {}

        # per-env random yaw applied at reset 
        self.random_yaw_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.random_yaw_quat[:, -1] = 1.0

        self.amp_observation_size = self.cfg.num_amp_observations * self.cfg.amp_observation_space
        self.amp_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.amp_observation_size,))
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space), device=self.device
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
                self._camera_eye_offset = torch.tensor([4.0, 0.0, -0.3], device=self.device)
            elif self.cfg.camera_type == "third-person":
                self._camera_target_offset = torch.tensor([1.0, 0.0, 0.5], device=self.device)
                self._camera_eye_offset = torch.tensor([-6.0, 0.0, 3.0], device=self.device)

            self._camera_smoothing_alpha = 0.1
            self._camera_eye_smooth: torch.Tensor | None = None
            self._camera_target_smooth: torch.Tensor | None = None

    def _setup_scene(self):
        self.robot = self.scene["robot"]

    def _update_camera(self):
        if self.cfg.camera_type == "none":
            return

        root_pos = self.robot.data.root_pos_w.torch[0]

        target_offset = transforms.quat_apply(self.random_yaw_quat, self._camera_target_offset.unsqueeze(0)).squeeze(0)
        eye_offset = transforms.quat_apply(self.random_yaw_quat, self._camera_eye_offset.unsqueeze(0)).squeeze(0)

        target = root_pos + target_offset
        eye = target + eye_offset

        if self._camera_target_smooth is None:
            self._camera_target_smooth = target.clone()
            self._camera_eye_smooth = eye.clone()
        else:
            alpha = self._camera_smoothing_alpha
            self._camera_target_smooth = alpha * target + (1.0 - alpha) * self._camera_target_smooth
            self._camera_eye_smooth = alpha * eye + (1.0 - alpha) * self._camera_eye_smooth

        eye_t = tuple(self._camera_eye_smooth.cpu().tolist())
        target_t = tuple(self._camera_target_smooth.cpu().tolist())

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
        self.robot.set_joint_position_target_index(target=self.actions)

        if self.render_enabled and self._sim_step_counter % self.cfg.decimation == 0:
            self._update_camera()

    def _reset_idx(self, env_ids: torch.Tensor):
        super()._reset_idx(env_ids)

        if self.render_enabled and self.cfg.camera_type != "none" and 0 in env_ids:
            self._camera_eye_smooth = None
            self._camera_target_smooth = None

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        num_samples = env_ids.shape[0]
        self.motion_ids[env_ids] = self.motions.sample_ids(num_samples)
        times = self.motions.library.sample_times(self.motion_ids[env_ids])
        self.motion_start_times[env_ids] = times
        frames = self.reference.frames_for_reset(self.motion_ids[env_ids], times)

        yaw = None
        if self.cfg.random_reset:
            yaw = transforms.random_yaw_orientation(num_samples, self.device)
            self.random_yaw_quat[env_ids] = yaw
        root_state, dof_pos, dof_vel = self._compute_robot_state(env_ids, frames, yaw)

        self._write_robot_state(
            env_ids,
            root_state,
            dof_pos,
            dof_vel,
        )

        done_envs = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        done_envs[env_ids] = True
        # self._resample_commands(dones=done_envs)

        self.amp_observation_buffer[env_ids] = self.reference.amp_seed(
            self.motion_ids[env_ids], times, self.random_yaw_quat[env_ids]
        )

    def _get_observations(self) -> dict:
        self._resample_commands()

        state = self.current_state
        local_frame = self.cfg.random_reset or self.render_enabled
        commands = policy_commands(state, self.commands, self.motions.has_goal[self.motion_ids], local_frame)
        policy_obs, style_obs = self.mdp.observations(state, commands, local_frame=local_frame)

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = style_obs

        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        return policy_obs

    def _get_rewards(self) -> torch.Tensor:
        context = self._step_context()
        total, self.extras["reward_terms_raw"], self.extras["reward_terms"] = self.mdp.rewards(context, self.motion_ids)
        self.extras["command_tracking"] = self.mdp.command_tracking(context, self.motion_ids)
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.mdp.died(self._step_context(), self.motion_ids), time_out

    def _step_context(self) -> StepContext:
        return StepContext(
            self.current_state,
            self.robot.data.joint_acc.torch,
            self.commands,
            self.actions,
            self.previous_actions,
            self._current_tracking_targets,
        )

    def _current_tracking_targets(self) -> TrackingTargets:
        """The reference pose at each environment's current clip time."""
        current_times = (
            self.motion_start_times
            + self.episode_length_buf.to(torch.float32) * self.step_dt
        )
        return self.reference.tracking_targets(self.motion_ids, current_times, self.random_yaw_quat)

    def _resample_commands(self, dones: torch.Tensor | None = None):
        """Resample commands for envs that have gone `command_resample_steps` steps since their
        last command sample or whose `dones` flag is set. Goals are placed relative to where each
        env's root is now, then stay fixed in world space."""

        if not self.cfg.resample_commands:
            return

        resample = self.episode_length_buf % (self.cfg.command_resample_steps + 1) == 0

        if dones is not None:
            resample = resample | dones

        resample_env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if resample_env_ids.numel() > 0:
            root_positions_xy = self.robot.data.body_pos_w.torch[resample_env_ids, self.ref_body_index, :2]
            self.commands[resample_env_ids] = self.motions.sample_commands(
                self.motion_ids[resample_env_ids], root_positions_xy
            )

            # a new goal or episode: progress toward the old goal no longer means anything
            self.mdp.reset_memory(resample_env_ids)

    def _compute_robot_state(
        self,
        env_ids: torch.Tensor,
        frames: ResetFrames,
        yaw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        root_state = self.default_root_state[env_ids].clone()
        root_state[:, 0:3] = frames.root_positions + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.1  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = frames.root_rotations
        root_state[:, 7:10] = frames.root_linear_velocities
        root_state[:, 10:13] = frames.root_angular_velocities

        if yaw is not None:
            root_state[:, 0:3] =  transforms.quat_apply(yaw, root_state[:, 0:3])
            root_state[:, 3:7] = transforms.quat_mul(yaw, root_state[:, 3:7])
            root_state[:, 7:10] = transforms.quat_apply(yaw, root_state[:, 7:10])
            root_state[:, 10:13] = transforms.quat_apply(yaw, root_state[:, 10:13])
        return root_state, frames.dof_positions, frames.dof_velocities
