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
class HumanoidAmpEnvCfg(DirectRLEnvCfg):
    """Humanoid AMP environment config (base class)."""

    # env
    episode_length_s = 10.0
    decimation = 2
    reward_type: str = "none" #tracking, joystick
    """Type of reward to be used in the environment.  

    * none: reward is one for all steps.
    * tracking: reward is based on tracking a reference motion.
    * joystick: reward is based on joystick input. For locomotion tasks only. 
    """

    # spaces
    observation_space = 81
    action_space = 28
    num_amp_observations = 2
    amp_observation_space = 81

    early_termination = True
    termination_height = 0.65

    deviation_termination: bool = False
    """Whether to terminate an episode early when the tracked pose deviates too far from the reference motion."""

    deviation_termination_threshold: float = 1.0
    """Weighted per-DOF squared-error threshold (see ``tracking_joint_weights``) above which an episode
    terminates when ``deviation_termination`` is enabled.
    """

    tracking_joint_weights: dict[str, float] = {}
    """Per-DOF weight used by the tracking reward (``reward_type == "tracking"``).

    Maps DOF name to its weight in the pose-tracking error. DOFs not present in this
    mapping default to a weight of 1.0.
    """

    tracking_reward_scale: float = 2.0
    """Scale applied to the weighted per-DOF tracking error: ``exp(-tracking_reward_scale * error)``."""

    motion_file: str = MISSING
    reference_body = "torso"
    reset_strategy = "random"  # default, random,
    """Strategy to be followed when resetting each environment (humanoid's pose and joint states).

    * default: pose and joint states are set to the initial state of the asset.
    * random: pose and joint states are set by sampling motions at random, uniform times.
    * random-start: pose and joint states are set by sampling motion at the start (time zero).
    """

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
class HumanoidAmpWalkEnvCfg(HumanoidAmpEnvCfg):
    motion_file = os.path.join(MOTIONS_DIR, "humanoid_walk.npz")
