# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from dataclasses import MISSING

from isaaclab_physx.physics import PhysxCfg

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets import HUMANOID_28_CFG

from moveitmoveit.utils.paths import MOTIONS_DIR


@configclass
class HumanoidSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the humanoid."""

    # humanoid
    humanoid: ArticulationCfg = HUMANOID_28_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )

    # ground
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # light
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


@configclass
class HumanoidEnvCfg(DirectRLEnvCfg):
    """Humanoid environment config (base class), shared by PPO and AMP.

    The environment always learns from motion capture data: it can (re)initialize episodes from
    a reference motion clip (``reset_strategy``) and/or reward tracking one (``reward_terms``).
    Which RL algorithm (PPO or AMP) drives training is decided entirely on the agent side; the
    environment itself doesn't distinguish between them.
    """

    # env
    episode_length_s = 10.0
    decimation = 2

    reward_terms: list[str] = ["joystick"]
    """Reward components to sum in ``_get_rewards``. Available terms:

    * tracking: reward for tracking a reference motion's DOF positions.
    * joystick: reward for tracking a commanded base-frame velocity.

    An empty list yields a constant reward of 1 for every step (e.g. for an AMP run driven purely
    by the style/discriminator reward).
    """

    # spaces
    observation_space = 81
    action_space = 28
    state_space = 0

    early_termination = True
    termination_height = 0.65

    deviation_termination: bool = False
    """Whether to terminate an episode early when the tracked pose deviates too far from the reference motion."""

    deviation_termination_threshold: float = 2.0
    """Weighted per-DOF squared-error threshold (see ``tracking_joint_weights``) above which an episode
    terminates when ``deviation_termination`` is enabled.
    """

    tracking_joint_weights: dict[str, float] = {}
    """Per-DOF weight used by the tracking reward (``"tracking" in reward_terms``).

    Maps DOF name to its weight in the pose-tracking error. DOFs not present in this
    mapping default to a weight of 1.0.
    """

    tracking_reward_scale: float = 2.0
    """Scale applied to the weighted per-DOF tracking error: ``exp(-tracking_reward_scale * error)``."""

    motion_files: list[str] = MISSING
    """Motion clip file paths. Sampled from uniformly at random each time a reference motion is drawn."""
    reference_body = "torso"
    reset_strategy = "random"  # default, random,
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

    standing_probability: float = 0.15
    """Probability that a sampled command is zeroed out (all-zero command), so the policy also
    learns to stand in place. Applied independently per env each time commands are (re)sampled."""

    # velocity command sampling ranges
    command_lin_vel_range: tuple[float, float] = (-1.5, 1.5)
    """Range (m/s) to sample the forward (base-frame x) linear velocity command from."""

    command_lat_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Range (m/s) to sample the lateral (base-frame y) linear velocity command from."""

    command_ang_vel_range: tuple[float, float] = (-1.0, 1.0)
    """Range (rad/s) to sample the yaw rate command from."""

    command_lin_vel_scale: float = 3.0
    """Scale applied to the squared linear velocity-command tracking error: ``exp(-command_lin_vel_scale * error)``."""

    command_ang_vel_scale: float = 3.0
    """Scale applied to the squared angular velocity-command tracking error: ``exp(-command_ang_vel_scale * error)``."""

    command_resampling_strategy: str = "interval"
    """When to resample each env's velocity command.

    * reset: resample only when the env resets (a new command each episode).
    * interval: also resample every ``command_resampling_interval_steps`` steps during the episode.
    """

    command_resampling_interval_steps: int = 150
    """Number of steps between command resamples when ``command_resampling_strategy == "interval"``."""

    def __post_init__(self):
        if "joystick" in self.reward_terms:
            self.observation_space += 3

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 60,
        render_interval=decimation,
        physics=PhysxCfg(gpu_found_lost_pairs_capacity=2**23, gpu_total_aggregate_pairs_capacity=2**23),
    )

    # scene
    scene: HumanoidSceneCfg = HumanoidSceneCfg(
        num_envs=4096, env_spacing=10.0, replicate_physics=True, clone_in_fabric=True
    )

    # robot
    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
        actuators={
            "body": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                velocity_limit_sim={
                    ".*": 100.0,
                },
            ),
        },
    )


@configclass
class HumanoidWalkEnvCfg(HumanoidEnvCfg):
    motion_files = [os.path.join(MOTIONS_DIR, "humanoid_walk.npz")]
