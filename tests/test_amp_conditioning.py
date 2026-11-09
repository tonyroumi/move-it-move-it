"""Conditional AMP: manifest labels, label-matched real batches, generated-sample labels and the
discriminator buffer, on CPU with a stub env (no simulator).

Run under IsaacLab's Python, which has torch: ./isaaclab.sh -p -m pytest tests
"""

import sys
from types import SimpleNamespace

import pytest
import torch
import yaml

from moveitmoveit import motion_set as motion_set_module
from moveitmoveit.manifest import CommandIndex, load_manifest
from moveitmoveit.motion_set import MotionSet


# --- manifests ----------------------------------------------------------------------------------


def write_manifest(tmp_path, motions: dict) -> str:
    path = tmp_path / "manifest.yaml"
    path.write_text(yaml.safe_dump({"motions": motions}, sort_keys=False))
    return str(path)


def gait(file: str, condition: int | None, sample_prob: float = 1.0, **extra) -> dict:
    motion = {"file": file, "sample_prob": sample_prob, "commands": {"lin_x": 1.0, "binary_key_0": float(condition or 0)}}
    if condition is not None:
        motion["style_condition"] = condition
    return motion | extra


def test_style_condition_labels_each_clip(tmp_path):
    specs = load_manifest(write_manifest(tmp_path, {"walk": gait("walk.npz", 0), "run": gait("run.npz", 1)}))
    assert [spec.style_condition for spec in specs] == [0, 1]


def test_a_manifest_without_style_condition_stays_unconditional(tmp_path):
    specs = load_manifest(write_manifest(tmp_path, {"walk": gait("walk.npz", None, zero_command_prob=0.5)}))
    assert specs[0].style_condition is None


@pytest.mark.parametrize(
    "motions, message",
    [
        ({"walk": gait("walk.npz", 0), "run": gait("run.npz", None)}, "every motion or by none"),
        ({"run": gait("run.npz", 1) | {"commands": {"binary_key_0": 0.0}}}, "fixed at 1"),
        ({"run": gait("run.npz", 1) | {"commands": {"binary_key_0": [0.0, 1.0]}}}, "fixed at 1"),
        ({"run": gait("run.npz", 1) | {"style_condition": 2}}, "expected one of"),
        ({"run": gait("run.npz", 1) | {"style_condition": True}}, "expected one of"),
        # a zeroed command without a standing block turns the key off: walk frames it has no clip for
        ({"run": gait("run.npz", 1, zero_command_prob=0.2)}, "no motion with a positive sample_prob"),
        ({"walk": gait("walk.npz", 0, sample_prob=0.0), "run": gait("run.npz", 1, zero_command_prob=0.2)}, "positive"),
    ],
)
def test_inconsistent_style_conditions_are_rejected(tmp_path, motions, message):
    with pytest.raises(ValueError, match=message):
        load_manifest(write_manifest(tmp_path, motions))


def test_a_zeroed_command_with_standing_needs_no_key_off_clip(tmp_path):
    # zeroed means standing, which belongs to no condition
    load_manifest(write_manifest(tmp_path, {"run": gait("run.npz", 1, zero_command_prob=0.2, standing={})}))


def test_the_shipped_walk_run_manifest_is_conditional():
    from moveitmoveit.utils.paths import CONFIGS_DIR

    specs = load_manifest(f"{CONFIGS_DIR}/walk_run_goal_manifest.yaml")
    assert {spec.name: spec.style_condition for spec in specs} == {"walk_to_goal": 0, "run_to_goal": 1}
    for spec in specs:
        assert spec.commands["binary_key_0"] == (float(spec.style_condition),) * 2
        assert "goal_x" in spec.commands and spec.standing is not None and "progress" in spec.rewards


# --- motion set ---------------------------------------------------------------------------------


@pytest.fixture
def motion_set(monkeypatch):
    """`MotionSet(specs)` over a stand-in MotionLibrary that loads no files."""
    monkeypatch.setattr(motion_set_module, "MotionLibrary", lambda files, device: SimpleNamespace(files=files))

    def build(motions: dict) -> MotionSet:
        from moveitmoveit.manifest import _parse_motion

        return MotionSet([_parse_motion(name, motion) for name, motion in motions.items()], "cpu")

    return build


def test_real_clips_are_drawn_only_from_the_requested_condition(motion_set):
    # several clips per condition, with sample_prob weights inside each
    motions = motion_set(
        {
            "walk_a": gait("walk_a.npz", 0, sample_prob=3.0),
            "run_a": gait("run_a.npz", 1, sample_prob=1.0),
            "walk_b": gait("walk_b.npz", 0, sample_prob=1.0),
            "run_b": gait("run_b.npz", 1, sample_prob=1.0),
        }
    )
    assert motions.num_style_conditions == 2

    torch.manual_seed(0)
    conditions = torch.tensor([0, 1] * 20_000)
    motion_ids = motions.sample_ids_for_conditions(conditions)

    assert torch.equal(motions.style_conditions[motion_ids], conditions)
    walk_ids = motion_ids[conditions == 0]
    assert set(walk_ids.tolist()) == {0, 2}
    assert (walk_ids == 0).float().mean().item() == pytest.approx(0.75, abs=0.02)
    run_ids = motion_ids[conditions == 1]
    assert (run_ids == 1).float().mean().item() == pytest.approx(0.5, abs=0.02)


def test_real_sampling_refuses_standing_or_unknown_conditions(motion_set):
    motions = motion_set({"walk": gait("walk.npz", 0), "run": gait("run.npz", 1)})
    for bad in (-1, 2):
        with pytest.raises(ValueError, match="must be in"):
            motions.sample_ids_for_conditions(torch.tensor([0, bad]))


def test_generated_conditions_follow_the_key_and_standing_has_none(motion_set):
    motions = motion_set({"walk": gait("walk.npz", 0), "run": gait("run.npz", 1)})
    commands = torch.zeros(4, len(CommandIndex))
    commands[:, CommandIndex.BINARY_KEY_0] = torch.tensor([0.0, 1.0, 1.0, 0.0])
    standing = torch.tensor([False, False, True, True])

    assert motions.generated_style_conditions(commands, standing).tolist() == [0, 1, -1, -1]


def test_an_unconditional_motion_set_has_no_condition_tables(motion_set):
    motions = motion_set({"walk": gait("walk.npz", None)})
    assert motions.num_style_conditions == 0
    with pytest.raises(RuntimeError):
        motions.sample_ids_for_conditions(torch.tensor([0]))


# --- generated-sample tracking ------------------------------------------------------------------


@pytest.fixture
def tracker(amp_module):
    from moveitmoveit.agents.amp.conditioning import StyleConditionTracker

    return lambda num_envs, num_amp_steps=2: StyleConditionTracker(num_envs, num_amp_steps, "cpu")


NO_RESET = torch.tensor([False, False])


def test_a_command_change_labels_the_next_transition_with_the_new_condition(tracker):
    t = tracker(2)
    assert t.update(torch.tensor([0, 0]), NO_RESET).tolist() == [True, True]
    # env 1's key turns on: with 2 frames, its next sample is already a transition taken under 1
    assert t.update(torch.tensor([0, 1]), NO_RESET).tolist() == [True, True]


def test_the_reset_step_is_dropped_and_the_next_one_kept(tracker):
    t = tracker(2)
    t.update(torch.tensor([1, 1]), NO_RESET)
    # env 0 was reset: its AMP observation is now the new episode's reference seed
    assert t.update(torch.tensor([1, 1]), torch.tensor([True, False])).tolist() == [False, True]
    assert t.update(torch.tensor([0, 1]), NO_RESET).tolist() == [True, True]


def test_standing_samples_are_never_kept(tracker):
    t = tracker(2)
    for _ in range(3):
        assert t.update(torch.tensor([-1, 0]), NO_RESET).tolist() == [False, True]


def test_a_longer_history_waits_until_it_holds_one_condition_only(tracker):
    t = tracker(1, num_amp_steps=3)
    no_reset = torch.tensor([False])
    assert t.update(torch.tensor([0]), no_reset).tolist() == [False]  # history still holds the seed
    assert t.update(torch.tensor([0]), no_reset).tolist() == [True]
    assert t.update(torch.tensor([1]), no_reset).tolist() == [False]  # one walk, one run transition
    assert t.update(torch.tensor([1]), no_reset).tolist() == [True]


# --- the AMP agent ------------------------------------------------------------------------------

NUM_ENVS = 4
AMP_DIM = 6
OBS_DIM = 5 + len(CommandIndex)
NUM_ACTIONS = 3


@pytest.fixture
def amp_module(monkeypatch):
    """`moveitmoveit.agents.amp.amp`, imported against a stand-in `isaaclab.envs`."""
    monkeypatch.setitem(sys.modules, "isaaclab", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "isaaclab.envs", SimpleNamespace(DirectRLEnv=object))
    for name in [name for name in sys.modules if name.startswith("moveitmoveit.agents")]:
        monkeypatch.delitem(sys.modules, name)
    from moveitmoveit.agents.amp import amp

    yield amp
    for name in [name for name in sys.modules if name.startswith("moveitmoveit.agents")]:
        sys.modules.pop(name, None)


class StubReference:
    """Real motion whose every feature equals its sample's condition (or 7 unconditionally),
    recording what each `style_batch` call asked for."""

    def __init__(self):
        self.calls = []

    def style_batch(self, num_samples, conditions=None):
        self.calls.append((num_samples, None if conditions is None else conditions.clone()))
        if conditions is None:
            return torch.full((num_samples, AMP_DIM), 7.0)
        return conditions.float().unsqueeze(-1).expand(-1, AMP_DIM).clone()


def stub_env(num_style_conditions: int):
    limits = torch.stack((-torch.ones(NUM_ACTIONS), torch.ones(NUM_ACTIONS)), dim=-1).unsqueeze(0)
    unwrapped = SimpleNamespace(
        device="cpu",
        step_dt=1 / 30,
        command_dim=len(CommandIndex),
        num_envs=NUM_ENVS,
        cfg=SimpleNamespace(num_amp_observations=2),
        robot=SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=SimpleNamespace(torch=limits))),
        reference=StubReference(),
        amp_observation_space=SimpleNamespace(shape=(AMP_DIM,)),
        num_style_conditions=num_style_conditions,
    )
    env = SimpleNamespace(
        observation_space=SimpleNamespace(shape=(OBS_DIM,)),
        action_space=SimpleNamespace(shape=(NUM_ACTIONS,)),
        unwrapped=unwrapped,
    )
    return env


def make_agent(amp_module, tmp_path, num_style_conditions: int):
    from moveitmoveit.utils.logger import Logger

    env = stub_env(num_style_conditions)
    cfg = amp_module.AMPCfg(
        class_type="AMP", num_transitions_per_env=4, disc_batch_size=64, disc_num_updates=1, style_reward_lambda=1.0
    )
    agent = amp_module.AMP(cfg=cfg, logger=Logger(backend="none", log_dir=str(tmp_path)))
    mlp = {"hidden_layers": [8], "activation": "relu"}
    agent.init(env, {"models": {"actor": mlp | {"noise_std_type": "fixed", "init_noise_std": 0.1}, "critic": mlp, "discriminator": mlp}})
    return agent, env


def step(agent, amp_obs, conditions=None, terminated=None, standing=None):
    """One env step into `agent`, with `amp_obs` (N, AMP_DIM) and the condition each env acted on."""
    agent.act(torch.zeros(NUM_ENVS, OBS_DIM))
    infos = {"amp_obs": amp_obs, "reward_terms": {}, "reward_terms_raw": {}}
    if standing is not None:
        infos["style_reward_off"] = standing
    if conditions is not None:
        infos["style_condition"] = conditions
    no_reset = torch.zeros(NUM_ENVS, dtype=torch.bool)
    agent.process_env_step(
        torch.zeros(NUM_ENVS, OBS_DIM),
        torch.zeros(NUM_ENVS),
        no_reset if terminated is None else terminated,
        no_reset,
        infos,
    )
    return infos


def record_inputs(module: torch.nn.Module) -> list[torch.Tensor]:
    """Every input `module` is called with from now on."""
    inputs = []
    module.register_forward_hook(lambda _module, args, _output: inputs.append(args[0].detach()))
    return inputs


def labeled_obs(values) -> torch.Tensor:
    """AMP observations whose every feature is the given per-env value, to spot rows later."""
    return torch.tensor(values, dtype=torch.float32).unsqueeze(-1).expand(-1, AMP_DIM).clone()


def test_conditional_buffer_rows_keep_their_own_label(amp_module, tmp_path):
    agent, _ = make_agent(amp_module, tmp_path, num_style_conditions=2)
    assert agent.discriminator.in_channels == AMP_DIM + 2

    # env 2 stands (-1) and env 3 is reset this step: neither is a generated sample of a condition
    step(
        agent,
        labeled_obs([0, 1, 5, 9]),
        conditions=torch.tensor([0, 1, -1, 1]),
        terminated=torch.tensor([False, False, False, True]),
        standing=torch.tensor([False, False, True, False]),
    )
    rows = agent._amp_observation_buffer.sample(256)
    assert len(agent._amp_observation_buffer) == 2
    observations, labels = rows[:, :-1], rows[:, -1]
    assert torch.equal(observations, labels.unsqueeze(-1).expand(-1, AMP_DIM))

    # the key changes between steps: each row carries the condition the env acted on at its step
    step(agent, labeled_obs([1, 0, 1, 1]), conditions=torch.tensor([1, 0, 1, 1]))
    assert len(agent._amp_observation_buffer) == 6
    rows = agent._amp_observation_buffer.sample(512)
    assert torch.equal(rows[:, :-1], rows[:, -1:].expand(-1, AMP_DIM))
    assert set(rows[:, -1].tolist()) == {0.0, 1.0}


def test_style_reward_scores_each_env_under_its_own_condition(amp_module, tmp_path):
    agent, _ = make_agent(amp_module, tmp_path, num_style_conditions=2)
    inputs = record_inputs(agent.discriminator)

    infos = step(agent, labeled_obs([0, 1, 1, 0]), conditions=torch.tensor([0, 1, -1, 1]))

    one_hot = inputs[-1][:, -2:]
    assert one_hot.tolist() == [[1, 0], [0, 1], [1, 0], [0, 1]]
    # the standing env has no condition: no style reward, even without a style_reward_off mask
    assert infos["reward_terms"]["style"][2] == 0
    assert (infos["reward_terms"]["style"][[0, 1, 3]] > 0).all()


def test_real_batches_match_the_generated_batch_condition_for_condition(amp_module, tmp_path):
    agent, env = make_agent(amp_module, tmp_path, num_style_conditions=2)
    inputs = record_inputs(agent.discriminator)

    # a mixed buffer, mostly walk
    for _ in range(3):
        step(agent, labeled_obs([0, 0, 0, 1]), conditions=torch.tensor([0, 0, 0, 1]))
    env.unwrapped.reference.calls.clear()
    inputs.clear()

    agent._update_discriminator()

    (num_samples, real_conditions), = env.unwrapped.reference.calls
    agent_input, real_input = inputs
    assert num_samples == 64 and real_conditions is not None
    assert 0 < real_conditions.float().mean() < 1  # mixed, not all one condition
    # generated and real one-hots are identical row for row: the label says nothing about realness
    assert torch.equal(agent_input[:, -2:], real_input[:, -2:])
    assert torch.equal(agent_input[:, -2:].argmax(-1), real_conditions)


def test_unconditional_amp_is_unchanged(amp_module, tmp_path):
    agent, env = make_agent(amp_module, tmp_path, num_style_conditions=0)
    assert agent.discriminator.in_channels == AMP_DIM
    assert agent._amp_observation_buffer._buffer.shape[-1] == AMP_DIM

    # every env reaches the buffer, reset and standing ones included; standing earns no style
    infos = step(
        agent,
        labeled_obs([0, 1, 2, 3]),
        terminated=torch.tensor([False, False, False, True]),
        standing=torch.tensor([False, False, True, False]),
    )
    assert len(agent._amp_observation_buffer) == NUM_ENVS
    assert infos["reward_terms"]["style"][2] == 0

    env.unwrapped.reference.calls.clear()
    agent._update_discriminator()
    assert env.unwrapped.reference.calls == [(64, None)]


def test_checkpoints_refuse_a_conditional_mismatch(amp_module, tmp_path):
    unconditional, _ = make_agent(amp_module, tmp_path / "a", num_style_conditions=0)
    unconditional.write_checkpoint(0, "old.pt")
    path = tmp_path / "a" / "checkpoints" / "old.pt"

    # a checkpoint from before conditional AMP has no num_style_conditions entry: still unconditional
    checkpoint = torch.load(path)
    del checkpoint["num_style_conditions"]
    torch.save(checkpoint, path)
    unconditional.load_checkpoint(str(path), "cpu")

    conditional, _ = make_agent(amp_module, tmp_path / "b", num_style_conditions=2)
    with pytest.raises(ValueError, match="train a new run"):
        conditional.load_checkpoint(str(path), "cpu")

    conditional.write_checkpoint(0, "new.pt")
    conditional.load_checkpoint(str(tmp_path / "b" / "checkpoints" / "new.pt"), "cpu")
