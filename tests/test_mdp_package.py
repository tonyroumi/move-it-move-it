"""The `moveitmoveit.mdp` package keeps the names the env and the motion reference import. No simulator."""

import importlib.util
import sys
import types

import torch

if importlib.util.find_spec("isaaclab") is None:
    # `utils.transforms`, the MDP terms, and the motion reference import quaternion ops from IsaacLab,
    # which is installed only beside Isaac Sim. Scriptable stand-ins are enough to import the package;
    # no term runs here.
    def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return vec

    def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return vec

    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        return q1

    def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
        return quat

    def random_yaw_orientation(num: int, device: str) -> torch.Tensor:
        return torch.zeros(num, 4, device=device)

    math = types.ModuleType("isaaclab.utils.math")
    math.quat_apply, math.quat_apply_inverse, math.quat_mul = quat_apply, quat_apply_inverse, quat_mul
    math.yaw_quat, math.random_yaw_orientation = yaw_quat, random_yaw_orientation
    sys.modules |= {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math,
    }

from moveitmoveit import mdp
from moveitmoveit.mdp import BodyState, ComposedMDP, StepContext, observation_dim, policy_commands
from moveitmoveit.mdp import composer, observations, rewards, terminations


def test_package_exports_the_names_env_and_reference_import():
    assert BodyState is composer.BodyState
    assert StepContext is composer.StepContext
    assert ComposedMDP is composer.ComposedMDP
    assert observation_dim is composer.observation_dim
    assert policy_commands is composer.policy_commands
    for name in ("BodyState", "ComposedMDP", "StepContext", "observation_dim", "policy_commands"):
        assert name in mdp.__all__


def test_each_kind_of_term_registers_from_its_own_module():
    assert mdp.OBSERVATIONS is observations.OBSERVATIONS
    assert mdp.REWARDS is rewards.REWARDS
    assert mdp.TERMINATIONS is terminations.TERMINATIONS

    assert list(observations.OBSERVATIONS) == [
        "dof_pos", "dof_vel", "root_height", "root_orientation", "root_lin_vel", "root_ang_vel", "key_body_offsets",
    ]
    # Registry order sets the order of logged reward terms and reward table columns.
    assert list(rewards.REWARDS) == [
        "motion_tracking", "heading_motion_tracking", "lin_vel_tracking", "yaw_vel_tracking", "target_hit",
        "line_following", "action_rate_l2", "joint_acc", "joint_vel", "progress", "root_lin_vel_l2",
        "root_ang_vel_l2", "upright", "support_centered", "standing_pose",
    ]
    assert list(terminations.TERMINATIONS) == ["deviation_from_motion", "unhealthy"]
    for name, term in observations.OBSERVATIONS.items():
        assert term.fn.__module__ == observations.__name__, name
    for name, term in rewards.REWARDS.items():
        assert term.fn.__module__ == rewards.__name__, name
    for name, fn in terminations.TERMINATIONS.items():
        assert fn.__module__ == terminations.__name__, name


def test_observation_dim_sums_the_selected_terms():
    assert observation_dim(["dof_pos", "root_height", "key_body_offsets"]) == 28 + 1 + 12


def test_motion_set_manifest_and_behavior_live_in_the_motion_package():
    from moveitmoveit.motion import behavior, manifest, motion_set, reference

    assert behavior.BehaviorManager.__module__ == "moveitmoveit.motion.behavior"
    assert manifest.load_manifest.__module__ == "moveitmoveit.motion.manifest"
    assert motion_set.MotionSet.__module__ == "moveitmoveit.motion.motion_set"
    assert reference.BodyState is BodyState

    for old in ("moveitmoveit.behavior", "moveitmoveit.manifest", "moveitmoveit.motion_set"):
        assert importlib.util.find_spec(old) is None, old
