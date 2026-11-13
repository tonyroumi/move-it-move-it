"""Termination terms take default arguments from @termination and per-motion overrides from the
manifest, like rewards. No simulator."""

import importlib
import sys
import textwrap
import types
from types import SimpleNamespace

import pytest
import torch


def _import_mdp_without_isaaclab() -> types.ModuleType:
    """`moveitmoveit.mdp`, imported against stand-ins for the IsaacLab ops its modules import.

    IsaacLab is installed only beside Isaac Sim, and importing it can start the simulator. The
    terminations run here never call a quaternion op, so the stand-ins only have to import; they
    leave sys.modules as they found it once the package is imported.
    """

    # typed like IsaacLab's, because `utils.transforms` scripts functions that call them
    def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        raise AssertionError("an IsaacLab stand-in was called")

    def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        raise AssertionError("an IsaacLab stand-in was called")

    def unused(*args, **kwargs):
        raise AssertionError("an IsaacLab stand-in was called")

    math = sys.modules.get("isaaclab.utils.math") or types.ModuleType("isaaclab.utils.math")
    functions = {"quat_apply": quat_apply, "quat_mul": quat_mul}
    for name in ("quat_apply", "quat_apply_inverse", "quat_mul", "yaw_quat", "random_yaw_orientation"):
        if not hasattr(math, name):
            setattr(math, name, functions.get(name, unused))
    stand_ins = {
        "isaaclab": types.ModuleType("isaaclab"),
        "isaaclab.utils": types.ModuleType("isaaclab.utils"),
        "isaaclab.utils.math": math,
    }
    added = [name for name in stand_ins if name not in sys.modules]
    sys.modules |= {name: stand_ins[name] for name in added}
    try:
        return importlib.import_module("moveitmoveit.mdp")
    finally:
        for name in added:
            del sys.modules[name]


mdp = _import_mdp_without_isaaclab()

from moveitmoveit.motion.manifest import load_manifest  # noqa: E402

NUM_DOFS = 2


def _manifest(tmp_path, motions: str):
    path = tmp_path / "manifest.yaml"
    path.write_text("motions:\n" + textwrap.indent(textwrap.dedent(motions), "  "))
    return load_manifest(str(path))


def _context(heights: list[float], dof_errors: list[float]) -> mdp.StepContext:
    """A step where each env's root sits at its height and its first DOF is off the reference by its error."""
    n = len(heights)
    root_positions = torch.zeros(n, 3)
    root_positions[:, 2] = torch.tensor(heights)
    dof_positions = torch.zeros(n, NUM_DOFS)
    dof_positions[:, 0] = torch.tensor(dof_errors)
    zeros = torch.zeros(n, 3)
    state = mdp.BodyState(dof_positions, torch.zeros(n, NUM_DOFS), root_positions, torch.zeros(n, 4), zeros, zeros, zeros)
    return mdp.StepContext(
        state=state,
        dof_accelerations=torch.zeros(n, NUM_DOFS),
        commands=torch.zeros(n, 9),
        actions=torch.zeros(n, NUM_DOFS),
        previous_actions=torch.zeros(n, NUM_DOFS),
        sample_targets=lambda: SimpleNamespace(dof_positions=torch.zeros(n, NUM_DOFS)),
    )


def _composed(specs) -> mdp.ComposedMDP:
    return mdp.ComposedMDP(specs, policy_obs=[], style_obs=[], device="cpu")


def test_the_decorator_registers_each_term_with_its_defaults():
    assert mdp.TERMINATIONS["deviation_from_motion"].defaults == {"threshold": 1.5}
    assert mdp.TERMINATIONS["unhealthy"].defaults == {"height_threshold": 0.52}


def test_a_name_only_termination_keeps_the_default_arguments(tmp_path):
    specs = _manifest(
        tmp_path,
        """
        walk:
          file: walk.npz
          termination: unhealthy
        dance:
          file: dance.npz
          termination: [unhealthy, deviation_from_motion]
        """,
    )
    assert specs[0].terminations == {"unhealthy": {}}
    assert specs[1].terminations == {"unhealthy": {}, "deviation_from_motion": {}}

    composed = _composed(specs)
    # heights either side of 0.52, and squared DOF errors either side of 1.5 (1.2^2 = 1.44, 1.3^2 = 1.69)
    c = _context(heights=[0.51, 0.53, 0.53, 0.53], dof_errors=[0.0, 0.0, 1.2, 1.3])
    assert composed.died(c, torch.tensor([0, 0, 0, 0])).tolist() == [True, False, False, False]
    c = _context(heights=[0.51, 0.53, 0.53, 0.53], dof_errors=[0.0, 0.0, 1.2, 1.3])
    assert composed.died(c, torch.tensor([1, 1, 1, 1])).tolist() == [True, False, False, True]


def test_a_motion_omitting_termination_gets_unhealthy_at_its_default(tmp_path):
    specs = _manifest(tmp_path, "walk:\n  file: walk.npz\n")
    assert specs[0].terminations == {"unhealthy": {}}
    c = _context(heights=[0.51, 0.53], dof_errors=[0.0, 0.0])
    assert _composed(specs).died(c, torch.tensor([0, 0])).tolist() == [True, False]


def test_manifest_arguments_reach_each_motions_terms(tmp_path):
    specs = _manifest(
        tmp_path,
        """
        walk:
          file: walk.npz
          termination: [unhealthy, deviation_from_motion]
        jab:
          file: jab.npz
          termination:
            unhealthy:
              height_threshold: 0.3
            deviation_from_motion:
              threshold: 4.0
        crouch:
          file: crouch.npz
          termination:
            unhealthy:
              height_threshold: 0.2
            deviation_from_motion:
        """,
    )
    assert specs[1].terminations == {"unhealthy": {"height_threshold": 0.3}, "deviation_from_motion": {"threshold": 4.0}}
    assert specs[2].terminations == {"unhealthy": {"height_threshold": 0.2}, "deviation_from_motion": {}}

    composed = _composed(specs)
    # each env at height 0.4 with a squared DOF error of 1.69: under walk's defaults both terms fire;
    # jab's lower height threshold and looser deviation threshold let it live; crouch keeps the default
    # deviation threshold, so it dies of deviation alone.
    motion_ids = torch.tensor([0, 1, 2])
    c = _context(heights=[0.4, 0.4, 0.4], dof_errors=[1.3, 1.3, 1.3])
    assert composed.died(c, motion_ids).tolist() == [True, False, True]

    # below jab's own height threshold, jab dies again
    c = _context(heights=[0.25, 0.25, 0.25], dof_errors=[0.0, 0.0, 0.0])
    assert composed.died(c, motion_ids).tolist() == [True, True, False]

    # past jab's own deviation threshold (2.1^2 = 4.41), jab dies again
    c = _context(heights=[0.6, 0.6, 0.6], dof_errors=[2.1, 2.1, 2.1])
    assert composed.died(c, motion_ids).tolist() == [True, True, True]


def test_an_unknown_termination_argument_is_rejected(tmp_path):
    specs = _manifest(
        tmp_path,
        """
        walk:
          file: walk.npz
          termination:
            unhealthy:
              threshold: 0.3
        """,
    )
    with pytest.raises(ValueError, match=r"walk\.termination\.unhealthy: unknown \['threshold'\]"):
        _composed(specs)


@pytest.mark.parametrize(
    "termination, message",
    [
        ("{unhealthy: {height_threshold: low}}", r"walk\.termination\.unhealthy\.height_threshold: expected a number"),
        ("{unhealthy: 0.3}", r"walk\.termination\.unhealthy: expected a mapping"),
        ("[unhealthy, {deviation_from_motion: {threshold: 2.0}}]", r"walk\.termination: expected a name"),
        ("3", r"walk\.termination: expected a name"),
    ],
)
def test_a_malformed_termination_is_rejected(tmp_path, termination, message):
    with pytest.raises(ValueError, match=message):
        _manifest(tmp_path, f"walk:\n  file: walk.npz\n  termination: {termination}\n")
