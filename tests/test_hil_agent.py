"""HIL warm starts every env on the tracking terms, then splits the envs into a PPO half on the
tracking terms and an AMP half on tracking, task and style rewards. No simulator."""

import importlib
import sys
import types
from types import SimpleNamespace

import pytest
import torch


def _import_agents_without_isaaclab() -> types.ModuleType:
    """`moveitmoveit.agents`, imported against a stand-in for the IsaacLab env class its modules
    name only in annotations. It leaves sys.modules as it found it once the package is imported."""
    envs = types.ModuleType("isaaclab.envs")
    envs.DirectRLEnv = object
    stand_ins = {"isaaclab": types.ModuleType("isaaclab"), "isaaclab.envs": envs}
    added = [name for name in stand_ins if name not in sys.modules]
    sys.modules |= {name: stand_ins[name] for name in added}
    try:
        return importlib.import_module("moveitmoveit.agents")
    finally:
        for name in added:
            del sys.modules[name]


agents = _import_agents_without_isaaclab()

from moveitmoveit.utils.logger import Logger  # noqa: E402

OBS_DIM, CMD_DIM, ACTION_DIM, AMP_DIM = 6, 3, 2, 4
NUM_TRANSITIONS = 2
MODELS = {
    name: {"hidden_layers": [8], "activation": "relu"} for name in ("critic", "discriminator")
} | {"actor": {"hidden_layers": [8], "activation": "relu", "noise_std_type": "fixed", "init_noise_std": 0.1}}


def _env(num_envs: int, reward_names: list[str]) -> SimpleNamespace:
    unwrapped = SimpleNamespace(
        step_dt=1.0 / 30.0,
        command_dim=CMD_DIM,
        num_envs=num_envs,
        device="cpu",
        robot=SimpleNamespace(
            data=SimpleNamespace(
                soft_joint_pos_limits=SimpleNamespace(torch=torch.tensor([[[-1.0, 1.0]] * ACTION_DIM]))
            )
        ),
        reference=SimpleNamespace(style_batch=lambda n: torch.randn(n, AMP_DIM)),
        amp_observation_space=SimpleNamespace(shape=(AMP_DIM,)),
        mdp=SimpleNamespace(reward_names=reward_names),
        sample_commands=True,
        sample_behaviors=True,
    )
    return SimpleNamespace(
        unwrapped=unwrapped,
        observation_space=SimpleNamespace(shape=(num_envs, OBS_DIM)),
        action_space=SimpleNamespace(shape=(num_envs, ACTION_DIM)),
    )


def _agent(tmp_path, num_envs: int, warm_start_steps: int, reward_names=None, **cfg):
    reward_names = reward_names or ["heading_motion_tracking", "progress", "joint_vel"]
    agent_cls, cfg_cls = agents.resolve_agent("HIL")
    agent = agent_cls(
        cfg=cfg_cls(
            **{
                "class_type": "HIL",
                "num_transitions_per_env": NUM_TRANSITIONS,
                "warm_start_steps": warm_start_steps,
                "discriminator_buffer_capacity": 1000,
                "disc_batch_size": 16,
                "disc_num_updates": 1,
                "goal_reward_lambda": 1.0,
                "style_reward_lambda": 2.0,
            }
            | cfg
        ),
        logger=Logger(backend="none", log_dir=str(tmp_path)),
    )
    env = _env(num_envs, reward_names)
    agent.init(env, {"models": MODELS})
    return agent, env


def _infos(num_envs: int, style_reward_off: torch.Tensor | None = None) -> dict:
    """Per-term rewards that tell the terms apart, and agent motion that names its env."""
    tracking = torch.full((num_envs,), 1.0)
    progress = torch.full((num_envs,), 10.0)
    joint_vel = torch.full((num_envs,), -100.0)
    return {
        "reward_terms": {
            "heading_motion_tracking": tracking,
            "progress": progress,
            "joint_vel": joint_vel,
            "task": tracking + progress + joint_vel,
        },
        "reward_terms_raw": {"heading_motion_tracking": tracking, "progress": progress, "joint_vel": joint_vel},
        "amp_obs": torch.arange(num_envs, dtype=torch.float32).unsqueeze(-1).repeat(1, AMP_DIM),
        "style_reward_off": torch.zeros(num_envs, dtype=torch.bool) if style_reward_off is None else style_reward_off,
    }


def _step(agent, num_envs: int, infos: dict | None = None) -> tuple[dict, torch.Tensor]:
    """One env step; returns the infos and the reward the agent stored for it."""
    infos = infos if infos is not None else _infos(num_envs)
    observations = torch.randn(num_envs, OBS_DIM)
    agent.act(observations)
    step = agent.storage.step
    agent.process_env_step(
        next_observations=observations,
        rewards=infos["reward_terms"]["task"].clone(),
        terminated=torch.zeros(num_envs, dtype=torch.bool),
        truncated=torch.zeros(num_envs, dtype=torch.bool),
        infos=infos,
    )
    return infos, agent.storage.rewards[step].view(-1)


def _style(agent, infos: dict) -> torch.Tensor:
    """The weighted style reward AMP would give each env."""
    return agent._style_reward(infos, infos["reward_terms"]["task"].shape)[1]


def _discriminator_params(agent) -> list[torch.Tensor]:
    return [p.detach().clone() for p in agent.discriminator.parameters()]


def test_warm_start_trains_every_env_on_tracking_alone(tmp_path):
    num_envs = 5
    agent, _ = _agent(tmp_path, num_envs, warm_start_steps=10**9)
    before = _discriminator_params(agent)

    for _ in range(NUM_TRANSITIONS):
        infos, rewards = _step(agent, num_envs)
        assert torch.equal(rewards, infos["reward_terms"]["heading_motion_tracking"])
        assert (infos["reward_terms"]["style"] == 0).all()
        assert (infos["reward_terms_raw"]["style"] == 0).all()

    assert agent.warm_starting
    assert len(agent._amp_observation_buffer) == 0

    agent.update()

    # PPO updated; the discriminator did not
    assert all(torch.equal(b, a) for b, a in zip(before, agent.discriminator.parameters()))


def test_split_puts_second_half_on_amp_and_odd_extra_env_on_ppo(tmp_path):
    num_envs = 5
    agent, _ = _agent(tmp_path, num_envs, warm_start_steps=0)
    ppo_half, amp_half = slice(0, 3), slice(3, 5)
    # standing envs on either half: the style reward is off there
    off = torch.tensor([True, False, False, True, False])
    infos = _infos(num_envs, off)
    style = _style(agent, infos)
    assert (style[[4]] != 0).all()

    infos, rewards = _step(agent, num_envs, infos)

    terms = infos["reward_terms"]
    tracking = terms["heading_motion_tracking"]
    task = terms["progress"] + terms["joint_vel"]
    assert not agent.warm_starting
    assert torch.equal(rewards[ppo_half], tracking[ppo_half])
    assert torch.allclose(rewards[amp_half], (tracking + task + style)[amp_half])
    assert rewards[3] == tracking[3] + task[3]  # standing: no style reward
    assert (terms["style"][ppo_half] == 0).all() and (infos["reward_terms_raw"]["style"][ppo_half] == 0).all()
    assert torch.allclose(terms["style"][amp_half], style[amp_half])


def test_discriminator_samples_agent_motion_from_the_amp_half_only(tmp_path):
    num_envs = 6
    agent, _ = _agent(tmp_path, num_envs, warm_start_steps=0)
    before = _discriminator_params(agent)

    for _ in range(NUM_TRANSITIONS):
        _step(agent, num_envs)

    buffer = agent._amp_observation_buffer
    assert len(buffer) == NUM_TRANSITIONS * 3
    # each agent-motion row names the env it came from; the AMP half is envs 3, 4 and 5
    assert set(buffer.sample(256)[:, 0].tolist()) == {3.0, 4.0, 5.0}

    agent.update()

    assert any(not torch.equal(b, a) for b, a in zip(before, agent.discriminator.parameters()))


def test_split_starts_once_warm_start_steps_have_completed(tmp_path):
    num_envs = 4
    agent, _ = _agent(tmp_path, num_envs, warm_start_steps=2 * num_envs, num_transitions_per_env=3)

    for expected_warm in (True, True, False):
        assert agent.warm_starting == expected_warm
        infos, rewards = _step(agent, num_envs)
        tracking = infos["reward_terms"]["heading_motion_tracking"]
        assert torch.equal(rewards[:2], tracking[:2])
        assert torch.equal(rewards[2:], tracking[2:]) == expected_warm

    assert not agent.warm_starting


def test_manifest_without_a_tracking_term_fails_at_startup(tmp_path):
    with pytest.raises(ValueError, match="no tracking term"):
        _agent(tmp_path, 4, warm_start_steps=0, reward_names=["progress", "joint_vel"])


def test_motion_tracking_counts_as_tracking(tmp_path):
    agent, _ = _agent(tmp_path, 4, warm_start_steps=0, reward_names=["motion_tracking", "heading_motion_tracking", "progress"])
    assert agent._tracking_terms == ["motion_tracking", "heading_motion_tracking"]
    assert agent._task_terms == ["progress"]


@pytest.mark.parametrize("resample_commands", [True, False])
@pytest.mark.parametrize("sample_behaviors", [True, False])
def test_command_and_behavior_sampling_reach_the_env(tmp_path, resample_commands, sample_behaviors):
    _, env = _agent(
        tmp_path, 4, warm_start_steps=0, resample_commands=resample_commands, sample_behaviors=sample_behaviors
    )
    assert env.unwrapped.sample_commands is resample_commands
    assert env.unwrapped.sample_behaviors is sample_behaviors


def test_policy_input_off_zeros_command_slots_and_keeps_the_size(tmp_path):
    observations = torch.randn(4, OBS_DIM)

    observing, _ = _agent(tmp_path, 4, warm_start_steps=0)
    assert torch.equal(observing._preprocess_observations(observations)[:, -CMD_DIM:], observations[:, -CMD_DIM:])

    blind, _ = _agent(tmp_path, 4, warm_start_steps=0, observe_commands=False)
    seen = blind._preprocess_observations(observations)
    assert seen.shape == observations.shape
    assert (seen[:, -CMD_DIM:] == 0).all()
    assert torch.equal(seen[:, :-CMD_DIM], observing._preprocess_observations(observations)[:, :-CMD_DIM])
    # the caller's observations are left as they were
    assert (observations[:, -CMD_DIM:] != 0).any()


def test_defaults_keep_commands_and_behaviors_on():
    cfg = agents.HILCfg(class_type="HIL")
    assert cfg.warm_start_steps == 0
    assert cfg.resample_commands and cfg.sample_behaviors and cfg.observe_commands


def test_amp_still_adds_style_to_every_env(tmp_path):
    num_envs = 4
    agent_cls, cfg_cls = agents.resolve_agent("AMP")
    agent = agent_cls(
        cfg=cfg_cls(class_type="AMP", num_transitions_per_env=NUM_TRANSITIONS, discriminator_buffer_capacity=1000, goal_reward_lambda=0.5),
        logger=Logger(backend="none", log_dir=str(tmp_path)),
    )
    agent.init(_env(num_envs, ["heading_motion_tracking", "progress", "joint_vel"]), {"models": MODELS})
    off = torch.tensor([True, False, False, False])
    infos = _infos(num_envs, off)
    style = _style(agent, infos)
    total = infos["reward_terms"]["task"].clone()

    infos, rewards = _step(agent, num_envs, infos)

    assert torch.allclose(rewards, 0.5 * total + style)
    assert style[0] == 0 and (style[1:] != 0).all()
    assert len(agent._amp_observation_buffer) == num_envs
