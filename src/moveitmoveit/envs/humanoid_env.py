# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
import warp as wp

from isaaclab.envs import DirectRLEnv

from moveitmoveit.utils import transforms
from moveitmoveit.utils.transforms import quaternion_to_tangent_and_normal
from moveitmoveit.motion import MotionLoader

from .humanoid_env_cfg import HumanoidEnvCfg


class HumanoidEnv(DirectRLEnv):
    """Humanoid environment for learning motion from motion-capture data."""

    cfg: HumanoidEnvCfg

    def __init__(self, cfg: HumanoidEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._setup_task()

        if "joystick" in self.cfg.reward_terms:
            self.commands = torch.zeros(self.num_envs, 3, device=self.sim.device)

    def _setup_scene(self):
        self.robot = self.scene["humanoid"]

    def _setup_task(self):
        """Position-target action interface and the shared motion library."""
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits.torch
        dof_lower_limits = soft_joint_pos_limits[0, :, 0]
        dof_upper_limits = soft_joint_pos_limits[0, :, 1]
        self.action_offset = 0.5 * (dof_upper_limits + dof_lower_limits)
        self.action_scale = dof_upper_limits - dof_lower_limits

        # load motion library (one or more clips)
        self._motion_loader = MotionLoader(motion_files=self.cfg.motion_files, device=self.device)

        # DOF and key body indexes
        key_body_names = ["right_hand", "left_hand", "right_foot", "left_foot"]
        self.ref_body_index = self.robot.data.body_names.index(self.cfg.reference_body)
        self.key_body_indexes = [self.robot.data.body_names.index(name) for name in key_body_names]
        self.motion_dof_indexes = self._motion_loader.get_dof_index(self.robot.data.joint_names)
        self.motion_ref_body_index = self._motion_loader.get_body_index([self.cfg.reference_body])[0]
        self.motion_key_body_indexes = self._motion_loader.get_body_index(key_body_names)

        # motion time (s) each env is currently tracking, set on reset and advanced via episode_length_buf
        self.motion_start_times = torch.zeros(self.num_envs, device=self.device)
        # motion clip (library index) each env is currently tracking, set on reset
        self.motion_clip_indexes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        # per-DOF weight for the tracking reward, ordered to match self.robot.data.joint_names
        self.tracking_joint_weights = torch.ones(len(self.robot.data.joint_names), device=self.device)
        for joint_name, weight in self.cfg.tracking_joint_weights.items():
            self.tracking_joint_weights[self.robot.data.joint_names.index(joint_name)] = weight

        # observation frame: base/local (direction-invariant) when joystick commands are part of the
        # reward, world-frame otherwise; used for both the live observation and the reference-motion
        # features so the two stay directly comparable (e.g. for an AMP discriminator)
        self._obs_fn = compute_joystick_obs if "joystick" in self.cfg.reward_terms else compute_obs

    @property
    def motion_dt(self) -> float:
        return self._motion_loader.dt

    def sample_reference_clip_times(self, num_samples: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample which clip, and a random time within it, for each of ``num_samples`` reference draws."""
        clip_indexes = self._motion_loader.sample_clip_indexes(num_samples)
        times = self._motion_loader.sample_times(num_samples, clip_indexes)
        return clip_indexes, times

    # joystick command sampling

    def _sample_commands(self, env_ids: torch.Tensor):
        num_samples = env_ids.shape[0]
        lin_vel_low, lin_vel_high = self.cfg.command_lin_vel_range
        lat_vel_low, lat_vel_high = self.cfg.command_lat_vel_range
        ang_vel_low, ang_vel_high = self.cfg.command_ang_vel_range
        self.commands[env_ids, 0] = torch.empty(num_samples, device=self.device).uniform_(lin_vel_low, lin_vel_high)
        self.commands[env_ids, 1] = torch.empty(num_samples, device=self.device).uniform_(lat_vel_low, lat_vel_high)
        self.commands[env_ids, 2] = torch.empty(num_samples, device=self.device).uniform_(ang_vel_low, ang_vel_high)

        if self.cfg.standing_probability > 0.0:
            zero_mask = torch.rand(num_samples, device=self.device) < self.cfg.standing_probability  # (num_samples,)
            self.commands[env_ids[zero_mask]] = 0.0

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

    def _pre_physics_step(self, actions: torch.Tensor):
        self.actions = actions.clone()

    def _apply_action(self):
        target = self.action_offset + self.action_scale * self.actions
        self.robot.set_joint_position_target_index(target=target)

    def _get_observations(self) -> torch.Tensor:
        obs = self._obs_fn(
            self.robot.data.joint_pos.torch,
            self.robot.data.joint_vel.torch,
            self.robot.data.body_pos_w.torch[:, self.ref_body_index],
            self.robot.data.body_quat_w.torch[:, self.ref_body_index],
            self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index],
            self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index],
            self.robot.data.body_pos_w.torch[:, self.key_body_indexes],
        )
        if "joystick" in self.cfg.reward_terms:
            obs = torch.hstack((obs, self.commands))
        return obs

    def _sample_reference_dof_positions(self) -> torch.Tensor:
        # current motion time (s) tracked per env: reset-time offset + elapsed steps in episode
        current_times = self.motion_start_times + self.episode_length_buf.to(torch.float32) * self.step_dt
        ref_dof_positions = self._motion_loader.sample(
            num_samples=self.num_envs,
            clip_indexes=self.motion_clip_indexes.cpu().numpy(),
            times=current_times.cpu().numpy(),
        )[0]
        return ref_dof_positions[:, self.motion_dof_indexes]

    def _get_rewards(self) -> torch.Tensor:
        if not self.cfg.reward_terms:
            return torch.ones((self.num_envs,), dtype=torch.float32, device=self.sim.device)

        reward = torch.zeros((self.num_envs,), dtype=torch.float32, device=self.sim.device)

        if "tracking" in self.cfg.reward_terms:
            ref_dof_positions = self._sample_reference_dof_positions()
            reward = reward + compute_tracking_reward(
                self.robot.data.joint_pos.torch,
                ref_dof_positions,
                self.tracking_joint_weights,
                self.cfg.tracking_reward_scale,
            )

        if "joystick" in self.cfg.reward_terms:
            root_rotation = self.robot.data.body_quat_w.torch[:, self.ref_body_index]
            local_lin_vel = transforms.quat_apply_inverse(
                root_rotation, self.robot.data.body_lin_vel_w.torch[:, self.ref_body_index]
            )
            local_ang_vel = transforms.quat_apply_inverse(
                root_rotation, self.robot.data.body_ang_vel_w.torch[:, self.ref_body_index]
            )
            reward = reward + compute_command_tracking_reward(
                local_lin_vel, local_ang_vel, self.commands, self.cfg.command_lin_vel_scale, self.cfg.command_ang_vel_scale
            )
            # resample commands (if due) after computing the reward for the command active during this transition
            self._resample_commands_mid_episode()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        if self.cfg.early_termination:
            died = self.robot.data.body_pos_w.torch[:, self.ref_body_index, 2] < self.cfg.termination_height
        else:
            died = torch.zeros_like(time_out)
        if self.cfg.deviation_termination:
            ref_dof_positions = self._sample_reference_dof_positions()
            deviation = compute_tracking_error(
                self.robot.data.joint_pos.torch, ref_dof_positions, self.tracking_joint_weights
            )
            died = died | (deviation > self.cfg.deviation_termination_threshold)
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            # Convert warp array to torch tensor if needed
            env_ids = (
                wp.to_torch(self.robot._ALL_INDICES)
                if isinstance(self.robot._ALL_INDICES, wp.array)
                else self.robot._ALL_INDICES
            )
        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        if self.cfg.reset_strategy == "default":
            root_state, joint_pos, joint_vel = self._reset_strategy_default(env_ids)
        elif self.cfg.reset_strategy.startswith("random"):
            start = "start" in self.cfg.reset_strategy
            root_state, joint_pos, joint_vel = self._reset_strategy_random(env_ids, start)
        else:
            raise ValueError(f"Unknown reset strategy: {self.cfg.reset_strategy}")

        self.robot.write_root_link_pose_to_sim_index(root_pose=root_state[:, :7], env_ids=env_ids)
        self.robot.write_root_com_velocity_to_sim_index(root_velocity=root_state[:, 7:], env_ids=env_ids)
        self.robot.write_joint_position_to_sim_index(position=joint_pos, env_ids=env_ids)
        self.robot.write_joint_velocity_to_sim_index(velocity=joint_vel, env_ids=env_ids)

        if "joystick" in self.cfg.reward_terms:
            self._sample_commands(env_ids)

    # reset strategies

    def _reset_strategy_default(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        default_root_pose = self.robot.data.default_root_pose.torch[env_ids].clone()
        default_root_vel = self.robot.data.default_root_vel.torch[env_ids].clone()
        default_root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_state = torch.cat([default_root_pose, default_root_vel], dim=-1)
        joint_pos = self.robot.data.default_joint_pos.torch[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel.torch[env_ids].clone()
        self.motion_start_times[env_ids] = 0.0
        self.motion_clip_indexes[env_ids] = 0
        return root_state, joint_pos, joint_vel

    def _reset_strategy_random(
        self, env_ids: torch.Tensor, start: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # sample which clip, and random motion times (or zeros if start is True), for each env
        num_samples = env_ids.shape[0]
        clip_indexes = self._motion_loader.sample_clip_indexes(num_samples)
        times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples, clip_indexes)
        self.motion_start_times[env_ids] = torch.tensor(times, dtype=torch.float32, device=self.device)
        self.motion_clip_indexes[env_ids] = torch.tensor(clip_indexes, dtype=torch.long, device=self.device)
        # sample random motions
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, clip_indexes=clip_indexes, times=times)

        # get root transforms (the humanoid torso)
        motion_torso_index = self._motion_loader.get_body_index(["torso"])[0]
        root_state = torch.cat(
            [
                self.robot.data.default_root_pose.torch[env_ids],
                self.robot.data.default_root_vel.torch[env_ids],
            ],
            dim=-1,
        ).clone()
        root_state[:, 0:3] = body_positions[:, motion_torso_index] + self.scene.env_origins[env_ids]
        root_state[:, 2] += 0.15  # lift the humanoid slightly to avoid collisions with the ground
        root_state[:, 3:7] = body_rotations[:, motion_torso_index]
        root_state[:, 7:10] = body_linear_velocities[:, motion_torso_index]
        root_state[:, 10:13] = body_angular_velocities[:, motion_torso_index]
        # get DOFs state
        dof_pos = dof_positions[:, self.motion_dof_indexes]
        dof_vel = dof_velocities[:, self.motion_dof_indexes]

        return root_state, dof_pos, dof_vel

    # env methods

    def collect_reference_motions(
        self,
        num_samples: int,
        current_times: np.ndarray | None = None,
        clip_indexes: np.ndarray | None = None,
    ) -> torch.Tensor:
        # sample which clip, and random motion times (or use the ones specified), per sample
        if clip_indexes is None:
            clip_indexes = self._motion_loader.sample_clip_indexes(num_samples)
        if current_times is None:
            current_times = self._motion_loader.sample_times(num_samples, clip_indexes)
        (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        ) = self._motion_loader.sample(num_samples=num_samples, clip_indexes=clip_indexes, times=current_times)
        return self._obs_fn(
            dof_positions[:, self.motion_dof_indexes],
            dof_velocities[:, self.motion_dof_indexes],
            body_positions[:, self.motion_ref_body_index],
            body_rotations[:, self.motion_ref_body_index],
            body_linear_velocities[:, self.motion_ref_body_index],
            body_angular_velocities[:, self.motion_ref_body_index],
            body_positions[:, self.motion_key_body_indexes],
        )


@torch.jit.script
def compute_tracking_error(
    dof_positions: torch.Tensor,
    ref_dof_positions: torch.Tensor,
    joint_weights: torch.Tensor,
) -> torch.Tensor:
    # dof_positions, ref_dof_positions: (N, num_dofs); joint_weights: (num_dofs,)
    return torch.sum(joint_weights * (dof_positions - ref_dof_positions) ** 2, dim=-1)  # (N,)


@torch.jit.script
def compute_tracking_reward(
    dof_positions: torch.Tensor,
    ref_dof_positions: torch.Tensor,
    joint_weights: torch.Tensor,
    reward_scale: float,
) -> torch.Tensor:
    weighted_error = compute_tracking_error(dof_positions, ref_dof_positions, joint_weights)  # (N,)
    return torch.exp(-reward_scale * weighted_error)


@torch.jit.script
def compute_obs(
    dof_positions: torch.Tensor,
    dof_velocities: torch.Tensor,
    root_positions: torch.Tensor,
    root_rotations: torch.Tensor,
    root_linear_velocities: torch.Tensor,
    root_angular_velocities: torch.Tensor,
    key_body_positions: torch.Tensor,
) -> torch.Tensor:
    obs = torch.cat(
        (
            dof_positions,
            dof_velocities,
            root_positions[:, 2:3],  # root body height
            quaternion_to_tangent_and_normal(root_rotations),
            root_linear_velocities,
            root_angular_velocities,
            (key_body_positions - root_positions.unsqueeze(-2)).view(key_body_positions.shape[0], -1),
        ),
        dim=-1,
    )
    return obs


@torch.jit.script
def compute_joystick_obs(
    dof_positions: torch.Tensor,  # (N, num_dofs)
    dof_velocities: torch.Tensor,  # (N, num_dofs)
    root_positions: torch.Tensor,  # (N, 3)
    root_rotations: torch.Tensor,  # (N, 4), xyzw
    root_linear_velocities: torch.Tensor,  # (N, 3), world frame
    root_angular_velocities: torch.Tensor,  # (N, 3), world frame
    key_body_positions: torch.Tensor,  # (N, num_key_bodies, 3)
) -> torch.Tensor:
    """Same layout as ``compute_obs``, but expressed in the robot's base frame instead of world frame, so the
    observation (and, for AMP, the style/discriminator features derived from it) is invariant to the commanded
    heading (used when ``"joystick" in cfg.reward_terms``)."""
    num_envs = root_positions.shape[0]

    # base-frame velocities: world-frame quantities rotated into the robot's own frame
    local_lin_vel = transforms.quat_apply_inverse(root_rotations, root_linear_velocities)  # (N, 3)
    local_ang_vel = transforms.quat_apply_inverse(root_rotations, root_angular_velocities)  # (N, 3)

    # key-body offsets expressed in the base frame
    key_offsets_w = key_body_positions - root_positions.unsqueeze(1)  # (N, num_key_bodies, 3)
    num_key_bodies = key_offsets_w.shape[1]
    key_rotations = root_rotations.unsqueeze(1).expand(num_envs, num_key_bodies, 4)
    local_key_positions = transforms.quat_apply_inverse(key_rotations, key_offsets_w).reshape(num_envs, -1)  # (N, num_key_bodies*3)

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
