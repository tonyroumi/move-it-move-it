import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv

from moveitmoveit.utils import transforms

from .behavior import LOCOMOTION, BehaviorManager, RandomKeyPresses, behavior_commands, reference_times, settled
from .env_cfg import MotionLearningEnvCfg
from .manifest import COMMAND_DIM
from .mdp import BodyState, ComposedMDP, StepContext, observation_dim, policy_commands
from .motion.reference import ReferenceMotions, ResetFrames, SkeletonMap, TrackingTargets
from .motion_metrics import MotionMetrics
from .motion_set import MotionSet


class MotionLearningEnv(DirectRLEnv):
    cfg: MotionLearningEnvCfg

    def __init__(self, cfg: MotionLearningEnvCfg, render_mode: str | None = None, **kwargs):
        cfg.observation_space = observation_dim(cfg.policy_obs) + COMMAND_DIM
        cfg.amp_observation_space = observation_dim(cfg.style_obs)
        super().__init__(cfg, render_mode, **kwargs)

        self.motions = MotionSet.from_manifest(cfg.motion_manifest, self.device)
        self.mdp = ComposedMDP(
            self.motions.specs, cfg.policy_obs, cfg.style_obs, self.device, joint_names=self.robot.data.joint_names
        )

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
        # the motion each env's rewards, terminations and tracking follow: its running behavior's,
        # else its locomotion motion's (base_motion_ids), which also owns its commands
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.base_motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # behaviors latched from raw key levels in commands[:, behavior_keys], the same way whether
        # training's random presses or play's keyboard wrote them
        self.behaviors = BehaviorManager(self.num_envs, self.motions.num_behaviors, self.device)
        self.behavior_phase = torch.zeros(self.num_envs, device=self.device)
        self.motion_metrics = MotionMetrics(self.motions, self.behaviors)
        """How often each motion is sampled and each behavior completes, for the training log."""
        self.key_presses = RandomKeyPresses(
            self.num_envs,
            self.motions.trigger_rate,
            self.motions.trigger_weights,
            max(1, round(self.cfg.trigger_max_hold_s / self.step_dt)),
            self.step_dt,
            self.device,
        )

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
        self.commands = torch.zeros(self.num_envs, self.command_dim, device=self.device)

        # Rewards and the AMP style gate read the mode of the observation the policy acted on.
        self.standing = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

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

        if self.cfg.camera_type == "third-person":
            root_quat = self.robot.data.root_quat_w.torch[:1]
            target_offset = transforms.quat_apply_yaw(root_quat, self._camera_target_offset.unsqueeze(0)).squeeze(0)
            eye_offset = transforms.quat_apply_yaw(root_quat, self._camera_eye_offset.unsqueeze(0)).squeeze(0)
        else:
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
        # each motion's random_start picks whether its clip clock starts at a random time or at 0
        clips = self.motions.sample_reset(num_samples)
        times = clips.times
        self.motion_ids[env_ids] = clips.motion_ids
        self.base_motion_ids[env_ids] = clips.base_motion_ids
        self.motion_start_times[env_ids] = clips.locomotion_start_times
        frames = self.reference.frames_for_reset(self.motion_ids[env_ids], times)
        self._reset_behaviors(env_ids, times)

        yaw = None
        if self.cfg.random_reset:
            yaw = transforms.random_yaw_orientation(num_samples, self.device)
            self.random_yaw_quat[env_ids] = yaw
        root_state, dof_pos, dof_vel = self._compute_robot_state(env_ids, frames, yaw)

        self._write_robot_state(env_ids, root_state, dof_pos, dof_vel)

        self.amp_observation_buffer[env_ids] = self.reference.amp_seed(
            self.motion_ids[env_ids], times, self.random_yaw_quat[env_ids]
        )

    def _reset_behaviors(self, env_ids: torch.Tensor, times: torch.Tensor):
        """An env reset into a behavior's clip at time `times` runs that behavior as if it had started
        `times` ago, so what it observes matches the pose it starts in, and returns to the locomotion
        motion `sample_reset` drew for it. Any other env starts in locomotion on the motion it reset into."""
        behavior = self.motions.motion_behavior[self.motion_ids[env_ids]]
        if self.cfg.resample_commands and self.motions.num_behaviors > 0:
            self.key_presses.reset(env_ids)
            self.commands[env_ids.unsqueeze(-1), self.motions.behavior_keys] = 0.0
        keys = self.commands[env_ids][:, self.motions.behavior_keys]
        # episode time restarts at 0, so a behavior `times` in started at -times
        self.behaviors.reset(env_ids, keys, behavior, -times)
        self.behavior_phase[env_ids] = 0.0

    def _update_behaviors(self):
        """Complete running behaviors at phase 1 or a stable posture, then start any whose key was
        just pressed, and point each env's motion at what it now runs."""
        time = self.episode_length_buf.to(torch.float32) * self.step_dt
        if self.cfg.resample_commands and self.motions.num_behaviors > 0:
            self.commands[:, self.motions.behavior_keys] = self.key_presses.step()

        duration = self.motions.library.duration[self.motion_ids]
        phase = self.behaviors.phase(time, duration)
        stable = settled(
            phase,
            self.robot.data.joint_vel.torch,
            self.motions.settle_min_phase[self.motion_ids],
            self.motions.settle_max_dof_speed[self.motion_ids],
        )
        complete = (phase >= 1.0) | stable
        self.behaviors.update(self.commands[:, self.motions.behavior_keys], complete, time)

        if self.motions.num_behaviors > 0:
            mode = self.behaviors.mode
            behavior_ids = self.motions.behavior_motion_ids[mode.clamp_min(0)]
            self.motion_ids = torch.where(mode != LOCOMOTION, behavior_ids, self.base_motion_ids)
        self.behavior_phase = self.behaviors.phase(time, self.motions.library.duration[self.motion_ids])

    def _get_observations(self) -> dict:
        self._resample_commands()
        self._update_behaviors()

        state = self.current_state
        local_frame = self.cfg.random_reset or self.render_enabled
        commands = policy_commands(state, self.commands, self.motions.has_goal[self.base_motion_ids], local_frame)
        commands = behavior_commands(commands, self.motions.behavior_keys, self.behaviors.mode, self.behavior_phase)
        # while a behavior runs, motion_ids is its motion, which can't declare standing: style stays on
        self.standing = self.motions.standing(self.motion_ids, commands)
        policy_obs, style_obs = self.mdp.observations(state, commands, local_frame=local_frame)

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = style_obs

        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        return policy_obs

    def _get_rewards(self) -> torch.Tensor:
        self.extras["standing"] = self.standing
        self.extras["style_reward_off"] = self._style_reward_off()
        context = self._step_context()
        total, self.extras["reward_terms_raw"], self.extras["reward_terms"] = self.mdp.rewards(
            context, self.motion_ids, self.standing
        )
        self.extras["command_tracking"] = self.mdp.command_tracking(context, self.motion_ids)
        return total

    def _style_reward_off(self) -> torch.Tensor:
        """envs that earn no AMP style reward this step"""
        return self.standing

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
        """The reference pose at each environment's current clip time. A running behavior's clip
        started with the behavior, at t_start; a locomotion clip with `loop` repeats."""
        time = self.episode_length_buf.to(torch.float32) * self.step_dt
        current_times = reference_times(
            time,
            self.behaviors,
            self.motion_start_times,
            self.motions.library.duration[self.motion_ids],
            self.motions.loop[self.motion_ids],
        )
        return self.reference.tracking_targets(self.motion_ids, current_times, self.random_yaw_quat)

    def _resample_commands(self, dones: torch.Tensor | None = None):
        """Resample commands every `command_resample_steps` steps or where `dones` is set."""
        if not self.cfg.resample_commands:
            return

        resample = self.episode_length_buf % (self.cfg.command_resample_steps + 1) == 0

        if dones is not None:
            resample = resample | dones

        resample_env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if resample_env_ids.numel() > 0:
            root_positions_xy = self.robot.data.body_pos_w.torch[resample_env_ids, self.ref_body_index, :2]
            self.commands[resample_env_ids] = self.motions.sample_commands(
                self.base_motion_ids[resample_env_ids], root_positions_xy
            )

            # a new goal or episode: progress toward the old goal no longer means anything
            self.mdp.reset_memory(resample_env_ids)

    def _compute_robot_state(
        self,
        env_ids: torch.Tensor,
        frames: ResetFrames,
        yaw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Root state, DOF positions and DOF velocities to write for `frames`, optionally yawed."""
        root_state = self.default_root_state[env_ids].clone()
        root_state[:, 0:3] = frames.root_positions + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.1  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = frames.root_rotations
        root_state[:, 7:10] = frames.root_linear_velocities
        root_state[:, 10:13] = frames.root_angular_velocities

        if yaw is not None:
            root_state[:, 0:3] = transforms.quat_apply(yaw, root_state[:, 0:3])
            root_state[:, 3:7] = transforms.quat_mul(yaw, root_state[:, 3:7])
            root_state[:, 7:10] = transforms.quat_apply(yaw, root_state[:, 7:10])
            root_state[:, 10:13] = transforms.quat_apply(yaw, root_state[:, 10:13])
        return root_state, frames.dof_positions, frames.dof_velocities
