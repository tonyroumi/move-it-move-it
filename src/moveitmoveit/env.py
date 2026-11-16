import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply, quat_apply_yaw, random_yaw_orientation

from moveitmoveit.env_cfg import MotionLearningEnvCfg
from moveitmoveit.mdp import MDP, observation_dim
from moveitmoveit.state import BodyState, Skeleton, StepContext
from moveitmoveit.mdp.commands import COMMAND_DIM
from moveitmoveit.motion.motion_manager import MotionManager
from moveitmoveit.motion.manifest import Manifest, load_manifest, manifest_files
from moveitmoveit.motion.library import MotionFrames, MotionLibrary
from moveitmoveit.utils.transforms import apply_random_yaw

class MotionLearningEnv(DirectRLEnv):
    cfg: MotionLearningEnvCfg

    def __init__(self, cfg: MotionLearningEnvCfg, render_mode: str | None = None, **kwargs):
        self.motion_library = MotionLibrary(manifest_files(cfg.manifest), cfg.sim.device)
        motion_specs = load_manifest(cfg.manifest, self.motion_library.duration.tolist())
        self.manifest = Manifest(motion_specs, cfg.sim.device)

        # the policy observation, commands, behavior one-hot, phase
        self.command_dim = COMMAND_DIM + self.manifest.num_behaviors + 1
        cfg.observation_space = observation_dim(cfg.policy_obs) + self.command_dim
        cfg.amp_observation_space = observation_dim(cfg.style_obs)
        super().__init__(cfg, render_mode, **kwargs)

        self.motion_manager = MotionManager(
            self.manifest,
            self.num_envs,
            self.device,
            self.step_dt,
        )

        self.mdp = MDP(
            policy_obs=cfg.policy_obs,
            style_obs=cfg.style_obs,
            specs=motion_specs,
            device=self.device,
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
            self.robot.data.joint_acc.torch,
            self.robot.data.body_pos_w.torch[:, self.skeleton.robot_root_body_index],
            self.robot.data.body_quat_w.torch[:, self.skeleton.robot_root_body_index],
            self.robot.data.body_lin_vel_w.torch[:, self.skeleton.robot_root_body_index],
            self.robot.data.body_ang_vel_w.torch[:, self.skeleton.robot_root_body_index],
            self.robot.data.body_pos_w.torch[:, self.skeleton.robot_key_body_indexes],
        )

    def reference_state(self) -> BodyState:
        time = self.episode_length_buf.to(torch.float32) * self.step_dt
        motion_ids = self.motion_manager.motion_ids
        frames = self.motion_library.frames(motion_ids, self.motion_manager.clip_times(time))
        return clip_state(frames, self.skeleton, self.random_yaw_quat if self.cfg.random_reset else None)

    def _setup_env(self):
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.skeleton = Skeleton(
            dof_indexes=self.motion_library.get_dof_index(self.robot.data.joint_names),
            root_body_index=self.motion_library.get_body_index(["torso"])[0],
            key_body_indexes=self.motion_library.get_body_index(key_body_names),
            robot_root_body_index=self.robot.data.body_names.index("torso"),
            robot_key_body_indexes=[self.robot.data.body_names.index(name) for name in key_body_names],
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

        self.actions = torch.zeros(self.num_envs, self.cfg.action_space, device=self.device)
        self.previous_actions = torch.zeros_like(self.actions)

        self.local_frame = self.cfg.random_reset or self.render_enabled

    def _setup_scene(self):
        self.robot = self.scene["robot"]

        if self.cfg.camera_type != "none":
            if self.cfg.camera_type == "facing":
                self._camera_target_offset = torch.tensor([2.5, 0.0, 0.0], device=self.device)
                self._camera_eye_offset = torch.tensor([3.0, 0.0, 0.0], device=self.device)
            elif self.cfg.camera_type == "third-person":
                self._camera_target_offset = torch.tensor([1.0, 0.0, 0.5], device=self.device)
                self._camera_eye_offset = torch.tensor([-6.0, 0.0, 3.0], device=self.device)

            self._camera_smoothing_alpha = 0.1
            self._camera_eye_smooth: torch.Tensor | None = None
            self._camera_target_smooth: torch.Tensor | None = None

    def _update_camera(self):
        if self.cfg.camera_type == "none":
            return

        root_pos = self.robot.data.root_pos_w.torch[0]

        if self.cfg.camera_type == "third-person":
            root_quat = self.robot.data.root_quat_w.torch[:1]
            target_offset = quat_apply_yaw(root_quat, self._camera_target_offset.unsqueeze(0)).squeeze(0)
            eye_offset = quat_apply_yaw(root_quat, self._camera_eye_offset.unsqueeze(0)).squeeze(0)
        else:
            target_offset = quat_apply(self.random_yaw_quat, self._camera_target_offset.unsqueeze(0)).squeeze(0)
            eye_offset = quat_apply(self.random_yaw_quat, self._camera_eye_offset.unsqueeze(0)).squeeze(0)

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

        # Reset camera smoothing when resetting environment
        if self.render_enabled and self.cfg.camera_type != "none" and 0 in env_ids:
            self._camera_eye_smooth = None
            self._camera_target_smooth = None

        self.actions[env_ids] = 0.0
        self.previous_actions[env_ids] = 0.0

        num_samples = env_ids.shape[0]

        # Reference state init
        motion_ids = self.manifest.sample_ids(num_samples)
        times = self.motion_library.sample_times(motion_ids)
        times = torch.where(self.manifest.random_start[motion_ids], times, torch.zeros_like(times))
        frames = self.motion_library.frames(motion_ids, times)
        self.motion_manager.reset(env_ids, motion_ids, times)

        yaw = None
        if self.cfg.random_reset:
            yaw = random_yaw_orientation(num_samples, self.device)
            self.random_yaw_quat[env_ids] = yaw
        root_state, dof_pos, dof_vel = self._compute_robot_state(env_ids, clip_state(frames, self.skeleton, yaw))

        self._write_robot_state(env_ids, root_state, dof_pos, dof_vel)

        self.amp_observation_buffer[env_ids] = self._amp_observations(motion_ids, times, yaw).view(
            num_samples, self.cfg.num_amp_observations, -1
        )

    def _get_observations(self) -> dict:
        self._resample_commands()

        time = self.episode_length_buf.to(torch.float32) * self.step_dt
        self.motion_manager.step(time, self.cfg.resample_commands)

        policy_obs, style_obs = self.mdp.observations(self._step_context(), self.local_frame)

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        self.amp_observation_buffer[:, 0] = style_obs

        self.extras["amp_obs"] = self.amp_observation_buffer.view(-1, self.amp_observation_size)

        return policy_obs

    def _get_rewards(self) -> torch.Tensor:
        context = self._step_context()
        self.extras["style_reward_weight"] = self.manifest.style_reward_weight(self.motion_manager.motion_ids)
        total, self.extras["reward_terms_raw"], self.extras["reward_terms"] = self.mdp.rewards(
            context, self.motion_manager.motion_ids
        )
        return total

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return self.mdp.died(self._step_context(), self.motion_manager.motion_ids), time_out

    def _step_context(self) -> StepContext:
        return StepContext(
            self.current_state,
            self.reference_state(),
            self.motion_manager.commands,
            self.previous_actions,
            self.actions,
            self.motion_manager.current_phase(),
            self.motion_manager.current_behaviors(),
            self.motion_manager.current_goals(),
        )

    def _resample_commands(self, dones: torch.Tensor | None = None):
        """Resample commands every `command_resample_steps` steps or where `dones` is set."""
        if not self.cfg.resample_commands:
            return

        resample = self.episode_length_buf % (self.cfg.command_resample_steps + 1) == 0

        if dones is not None:
            resample = resample | dones

        resample_env_ids = resample.nonzero(as_tuple=False).squeeze(-1)
        if resample_env_ids.numel() > 0:
            self.motion_manager.resample_commands(resample_env_ids, self.current_state)

            # a new goal or episode: progress toward the old goal no longer means anything
            self.mdp.reset_memory(resample_env_ids)

    def _compute_robot_state(
        self,
        env_ids: torch.Tensor,
        state: BodyState,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_state = self.default_root_state[env_ids].clone()
        root_state[:, 0:3] = state.root_positions + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.1  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = state.root_rotations
        root_state[:, 7:10] = state.root_linear_velocities
        root_state[:, 10:13] = state.root_angular_velocities

        return root_state, state.dof_positions, state.dof_velocities

    def style_batch(self, num_samples: int) -> torch.Tensor:
        """Sample reference style observations with history flattened per sample."""
        motion_ids = self.manifest.sample_ids(num_samples)
        times = self.motion_library.sample_times(motion_ids)
        yaw = random_yaw_orientation(num_samples, self.device) if self.cfg.random_reset else None
        return self._amp_observations(motion_ids, times, yaw)

    def _amp_observations(
        self, motion_ids: torch.Tensor, times: torch.Tensor, yaw: torch.Tensor | None
    ) -> torch.Tensor:
        """Build style observations from reference motion history."""
        num_samples = motion_ids.shape[0]
        history_length = self.cfg.num_amp_observations

        # the agent's history advances once per env step, so the reference's steps are step_dt apart too
        frames = self.motion_library.history(motion_ids, times, history_length, self.step_dt)
        if yaw is not None:
            yaw = yaw.repeat_interleave(history_length, dim=0)
        state = clip_state(frames, self.skeleton, yaw)

        observations = self.mdp.style_obs(state, self.local_frame)
        return observations.reshape(num_samples, -1)


def clip_state(frames: MotionFrames, skeleton: Skeleton, yaw: torch.Tensor | None = None) -> BodyState:
    """Map clip frames into robot-ordered state, optionally yawed about the clip root."""
    body_positions = frames.body_positions
    body_rotations = frames.body_rotations
    body_linear_velocities = frames.body_linear_velocities
    body_angular_velocities = frames.body_angular_velocities
    if yaw is not None:
        body_positions, body_rotations, body_linear_velocities, body_angular_velocities = apply_random_yaw(
            yaw,
            skeleton.root_body_index,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        )

    root = skeleton.root_body_index
    dof_positions = frames.dof_positions[:, skeleton.dof_indexes]
    return BodyState(
        dof_positions=dof_positions,
        dof_velocities=frames.dof_velocities[:, skeleton.dof_indexes],
        dof_accelerations=torch.zeros_like(dof_positions),  # clips carry none; no term reads them
        root_positions=body_positions[:, root],
        root_rotations=body_rotations[:, root],
        root_linear_velocities=body_linear_velocities[:, root],
        root_angular_velocities=body_angular_velocities[:, root],
        key_body_positions=body_positions[:, skeleton.key_body_indexes],
    )
