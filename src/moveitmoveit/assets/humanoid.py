"""Configuration for a Humanoid robot."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

from moveitmoveit.utils.paths import MODEL_DESCRIPTIONS_DIR

##
# Configuration
##

HUMANOID_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{MODEL_DESCRIPTIONS_DIR}/humanoid.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            max_depenetration_velocity=10.0,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.001,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.34),
        joint_pos={".*": 0.0},
    ),
    actuators={
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            stiffness=None,
            damping=None,
            effort_limit=None,
            velocity_limit_sim={
                ".*": 100.0,
            },
        ),
    },
)
"""Configuration for a Humanoid robot."""