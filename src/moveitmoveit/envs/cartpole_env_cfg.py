from __future__ import annotations

import isaaclab.sim as sim_utils

from isaaclab_newton.physics import KaminoSolverCfg, MJWarpSolverCfg, NewtonCfg
from isaaclab_ovphysx.physics import OvPhysxCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.utils import PresetCfg
from isaaclab_assets.robots.cartpole import CARTPOLE_CFG


@configclass
class CartpolePhysicsCfg(PresetCfg):
    default: PhysxCfg = PhysxCfg()
    physx: PhysxCfg = PhysxCfg()

    newton_mjwarp: NewtonCfg = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            njmax=5,
            nconmax=3,
            cone="pyramidal",
            impratio=1,
            integrator="implicitfast",
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )

    newton_kamino: NewtonCfg = NewtonCfg(
        solver_cfg=KaminoSolverCfg(
            integrator="moreau",
            use_collision_detector=True,
            sparse_jacobian=True,
            constraints_alpha=0.1,
            padmm_max_iterations=100,
            padmm_primal_tolerance=1e-4,
            padmm_dual_tolerance=1e-4,
            padmm_compl_tolerance=1e-4,
            padmm_rho_0=0.05,
            padmm_eta=1e-5,
            padmm_use_acceleration=True,
            padmm_warmstart_mode="containers",
            padmm_contact_warmstart_method="geom_pair_net_force",
            padmm_use_graph_conditionals=False,
            collision_detector_pipeline="unified",
            collision_detector_max_contacts_per_pair=8,
        ),
        num_substeps=1,
        debug_mode=False,
        use_cuda_graph=True,
    )

    ovphysx: OvPhysxCfg = OvPhysxCfg()


@configclass
class CartpoleSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the cartpole."""

    # cartpole
    cartpole: ArticulationCfg = CARTPOLE_CFG.replace(
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
class CartpoleEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 5.0

    action_scale = 100.0

    action_space = 1
    observation_space = 4
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics=CartpolePhysicsCfg(),
    )

    # scene
    scene: CartpoleSceneCfg = CartpoleSceneCfg(
        num_envs=4096,
        env_spacing=4.0,
        replicate_physics=True,
        clone_in_fabric=True,
    )

    cart_dof_name = "slider_to_cart"
    pole_dof_name = "cart_to_pole"

    # reset
    max_cart_pos = 3.0
    initial_pole_angle_range = [-0.25, 0.25]

    # rewards
    rew_scale_alive = 1.0
    rew_scale_terminated = -2.0
    rew_scale_pole_pos = -1.0
    rew_scale_cart_vel = -0.01
    rew_scale_pole_vel = -0.005