"""The tracking rewards take only their scales from @reward and the manifest; their component weights
are fixed by the function signature. No simulator."""

import inspect
from types import SimpleNamespace

import pytest
import torch

from test_termination_args import _composed, _manifest, mdp

from moveitmoveit.mdp import rewards

TRACKING = ["motion_tracking", "heading_motion_tracking"]
SCALES = {
    "dof_scale": 2.0,
    "key_body_scale": 5.0,
    "dof_vel_scale": 0.05,
    "root_vel_scale": 2.0,
    "root_ang_vel_scale": 0.5,
}
WEIGHTS = {
    "dof_weight": 0.25,
    "key_body_weight": 0.20,
    "dof_vel_weight": 0.20,
    "root_vel_weight": 0.25,
    "root_ang_vel_weight": 0.10,
}


def _signature_defaults(fn) -> dict[str, float]:
    return {
        name: param.default
        for name, param in inspect.signature(fn).parameters.items()
        if param.default is not inspect.Parameter.empty
    }


def _body(n: int, seed: int) -> mdp.BodyState:
    g = torch.Generator().manual_seed(seed)

    def rand(*shape):
        return torch.randn(*shape, generator=g)

    return mdp.BodyState(rand(n, 3), rand(n, 3), rand(n, 3), rand(n, 4), rand(n, 3), rand(n, 3), rand(n, 4, 3))


def _track(actual, target, scale):
    return torch.exp(-scale * (actual - target).square().mean(dim=-1))


@pytest.mark.parametrize("name", TRACKING)
def test_the_decorator_holds_only_the_scales(name):
    assert rewards.REWARDS[name].defaults == SCALES


@pytest.mark.parametrize("name", TRACKING)
def test_the_signature_holds_every_default(name):
    assert _signature_defaults(rewards.REWARDS[name].fn) == SCALES | WEIGHTS


@pytest.mark.parametrize("name", TRACKING)
def test_calling_with_only_the_scales_uses_the_signature_weights(name, monkeypatch):
    # Heading-frame rotation needs IsaacLab's quaternion ops; plain root-relative offsets and world-frame
    # vectors keep the reward's sum of weighted components without them.
    monkeypatch.setattr(rewards, "_heading_frame_offsets", lambda bodies, roots, rotations: bodies - roots.unsqueeze(1))
    monkeypatch.setattr(rewards, "_heading_frame_vectors", lambda vectors, rotations: vectors)

    state, targets = _body(5, seed=0), _body(5, seed=1)
    c = SimpleNamespace(state=state, targets=targets)
    offsets = (state.key_body_positions - state.root_positions.unsqueeze(1)).flatten(1)
    ref_offsets = (targets.key_body_positions - targets.root_positions.unsqueeze(1)).flatten(1)
    expected = (
        WEIGHTS["dof_weight"] * _track(state.dof_positions, targets.dof_positions, SCALES["dof_scale"])
        + WEIGHTS["key_body_weight"] * _track(offsets, ref_offsets, SCALES["key_body_scale"])
        + WEIGHTS["dof_vel_weight"] * _track(state.dof_velocities, targets.dof_velocities, SCALES["dof_vel_scale"])
        + WEIGHTS["root_vel_weight"]
        * _track(state.root_linear_velocities, targets.root_linear_velocities, SCALES["root_vel_scale"])
        + WEIGHTS["root_ang_vel_weight"]
        * _track(state.root_angular_velocities, targets.root_angular_velocities, SCALES["root_ang_vel_scale"])
    )

    fn = rewards.REWARDS[name].fn
    # the composer passes exactly the decorator's arguments
    torch.testing.assert_close(fn(c, **rewards.REWARDS[name].defaults), expected)
    torch.testing.assert_close(fn(c), expected)


@pytest.mark.parametrize("name", TRACKING)
def test_the_composer_passes_only_the_scales(name, tmp_path):
    specs = _manifest(tmp_path, f"walk:\n  file: walk.npz\n  rewards:\n    {name}: {{weight: 1.0, dof_scale: 3.0}}\n")
    composed = _composed(specs)
    assert set(composed.reward_kwargs[name]) == set(SCALES)
    assert composed.reward_kwargs[name]["dof_scale"].tolist() == [3.0]


@pytest.mark.parametrize("name", TRACKING)
def test_a_manifest_cannot_override_a_weight(name, tmp_path):
    specs = _manifest(tmp_path, f"walk:\n  file: walk.npz\n  rewards:\n    {name}: {{weight: 1.0, dof_weight: 0.5}}\n")
    with pytest.raises(ValueError, match=rf"walk\.rewards\.{name}: unknown \['dof_weight'\]"):
        _composed(specs)
