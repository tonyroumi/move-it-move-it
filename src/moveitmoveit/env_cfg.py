from typing import Literal

import isaaclab.sim as sim_utils

from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_assets import HUMANOID_28_CFG


@configclass
class HumanoidSceneCfg(InteractiveSceneCfg):
    """Scene layout — ground, lighting, the one humanoid. No per-task variants."""

    robot: ArticulationCfg = HUMANOID_28_CFG.replace(prim_path="/World/envs/env_.*/Robot")

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
class MotionLearningEnvCfg(DirectRLEnvCfg):
    # sim / scene
    decimation: int = 2
    episode_length_s: float = 10.0

    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    scene: HumanoidSceneCfg = HumanoidSceneCfg(num_envs=4096, env_spacing=10.0)

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

    action_space: int = 28
    state_space: int = 0
    num_amp_observations = 2

    policy_obs: tuple[str, ...] = ("dof_pos", "dof_vel", "root_height", "root_orientation", "root_lin_vel", "root_ang_vel", "key_body_offsets")
    style_obs: tuple[str, ...] = ("dof_pos", "dof_vel", "root_height", "root_orientation", "root_lin_vel", "root_ang_vel", "key_body_offsets")

    # derived from the selections above when the env is built
    observation_space: int = 0
    amp_observation_space = 0

    motion_manifest: str = ""

    # environment flags
    random_reset: bool = True
    command_resample_steps: int = 150
    # off while play.py drives the commands from the keyboard
    resample_commands: bool = True

    # rendering
    camera_type: Literal["facing", "third-person", "none"] = "none"
