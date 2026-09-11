from __future__ import annotations

import torch
import numpy as np 

from isaaclab.utils.math import quat_apply_inverse

from moveitmoveit.utils.transforms import quaternion_to_tangent_and_normal

from .humanoid_amp_env import HumanoidAmpEnv
from .joystick_humanoid_amp_env_cfg import JoystickHumanoidAmpEnvCfg


class JoystickHumanoidAmpEnv(HumanoidAmpEnv):
    """Humanoid AMP environment with a command-conditioned joystick locomotion task.

    Each env is given a randomly sampled velocity command (forward, lateral, yaw rate),
    resampled on reset and, optionally, at a fixed step interval during the episode (see
    ``cfg.command_resampling_strategy``), and rewarded for tracking it. The policy
    observation expresses linear/angular velocity and key-body offsets relative to the
    robot's base frame so it does not depend on the robot's global position. The AMP style
    observation used by the discriminator is left untouched (world-frame, no command) so it
    stays comparable to the reference motion features produced by ``collect_reference_motions``.
    """

    cfg: JoystickHumanoidAmpEnvCfg

    def __init__(self, cfg: JoystickHumanoidAmpEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)

    def _sample_commands(self, env_ids: torch.Tensor):
        num_samples = env_ids.shape[0]
        lin_vel_low, lin_vel_high = self.cfg.command_lin_vel_range
        lat_vel_low, lat_vel_high = self.cfg.command_lat_vel_range
        ang_vel_low, ang_vel_high = self.cfg.command_ang_vel_range
        self.commands[env_ids, 0] = torch.empty(num_samples, device=self.device).uniform_(lin_vel_low, lin_vel_high)
        self.commands[env_ids, 1] = torch.empty(num_samples, device=self.device).uniform_(lat_vel_low, lat_vel_high)
        self.commands[env_ids, 2] = torch.empty(num_samples, device=self.device).uniform_(ang_vel_low, ang_vel_high)

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device)
        self._sample_commands(env_ids)

    def _resample_commands_mid_episode(self):
        if self.cfg.command_resampling_strategy == "reset":
            return
        elif self.cfg.command_resampling_strategy == "interval":
            due = (self.episode_length_buf > 0) & (
                self.episode_length_buf % self.cfg.command_resampling_interval_steps == 0
            )
            env_ids = due.nonzero(as_tuple=False).squeeze(-1)
            if len(env_ids) > 0:
                self._sample_commands(env_ids)
        else:
            raise ValueError(f"Unknown command resampling strategy: {self.cfg.command_resampling_strategy}")

    def _get_observations(self) -> torch.Tensor:
        # updates self.extras["amp_obs"] from the (unmodified, world-frame) AMP style observation
        obs = compute_obs(
            self.robot.data.joint_pos.torch,
            self.robot.data.joint_vel.torch,
            self.robot.data.body_pos_w.torch[:, self.ref_body_index],
            self.robot.data.body_quat_w.torch[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index],
            self.robot.data.body_pos_w.torch[:, self.key_body_indexes],
        )

        # update AMP observation history
        for i in reversed(range(self.cfg.num_amp_observations - 1)):
            self.amp_observation_buffer[:, i + 1] = self.amp_observation_buffer[:, i]
        # build AMP observation
        self.amp_observation_buffer[:, 0] = obs.clone()
        self.extras = {"amp_obs": self.amp_observation_buffer.view(-1, self.amp_observation_size)}

        obs = torch.hstack((obs, self.commands))
        return obs

    def _get_rewards(self) -> torch.Tensor:
        root_rotation = self.robot.data.body_quat_w.torch[:, self.ref_body_index]
        local_lin_vel = quat_apply_inverse(root_rotation, self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index])
        local_ang_vel = quat_apply_inverse(root_rotation, self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index])
        reward = compute_command_tracking_reward(
            local_lin_vel, local_ang_vel, self.commands, self.cfg.command_lin_vel_scale, self.cfg.command_ang_vel_scale
        )
        # resample commands (if due) after computing the reward for the command active during this transition
        self._resample_commands_mid_episode()
        return reward

    def collect_reference_motions(self, num_samples: int, current_times: np.ndarray | None = None) -> torch.Tensor:
        # sample random motion times (or use the one specified)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples)
        times = (
            np.expand_dims(current_times, axis=-1)
            - self._motion_loader.dt * np.arange(0, self.cfg.num_amp_observations)
        ).flatten()
        # get motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, times=times)
        # compute AMP observation
        amp_observation = compute_obs(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )
        return amp_observation.view(-1, self.amp_observation_size)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,  # (N, num_dofs)
    dof_velocities: torch.Tensor,  # (N, num_dofs)
    root_positions: torch.Tensor,  # (N, 3)
    root_rotations: torch.Tensor,  # (N, 4), xyzw
    root_linear_velocities: torch.Tensor,  # (N, 3), world frame
    root_angular_velocities: torch.Tensor,  # (N, 3), world frame
    key_body_positions: torch.Tensor,  # (N, num_key_bodies, 3)
) -> torch.Tensor:
    num_envs = root_positions.shape[0]

    # base-frame velocities: world-frame quantities rotated into the robot's own frame
    local_lin_vel = quat_apply_inverse(root_rotations, root_linear_velocities)  # (N, 3)
    local_ang_vel = quat_apply_inverse(root_rotations, root_angular_velocities)  # (N, 3)

    # key-body offsets expressed in the base frame
    key_offsets_w = key_body_positions - root_positions.unsqueeze(1)  # (N, num_key_bodies, 3)
    num_key_bodies = key_offsets_w.shape[1]
    key_rotations = root_rotations.unsqueeze(1).expand(num_envs, num_key_bodies, 4)
    local_key_positions = quat_apply_inverse(key_rotations, key_offsets_w).reshape(num_envs, -1)  # (N, num_key_bodies*3)

    return torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            local_lin_vel,
            local_ang_vel,
            local_key_positions,
        ),
        dim=-1,
    )


@torch.jit.script
def compute_command_tracking_reward(
    local_linear_velocity: torch.Tensor,  # (N, 3), base-frame linear velocity
    local_angular_velocity: torch.Tensor,  # (N, 3), base-frame angular velocity
    commands: torch.Tensor,  # (N, 3): [lin_vel_x, lat_vel_y, yaw_rate]
    lin_scale: float,
    ang_scale: float,
) -> torch.Tensor:
    lin_vel_error = torch.sum((commands[:, :2] - local_linear_velocity[:, :2]) ** 2, dim=-1)  # (N,)
    ang_vel_error = (commands[:, 2] - local_angular_velocity[:, 2]) ** 2  # (N,)
    reward = torch.exp(
        -lin_scale * lin_vel_error
        -ang_scale * ang_vel_error
    )
    return reward
