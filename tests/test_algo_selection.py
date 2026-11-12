"""`--algo` selects the agent config, and that config builds the matching trainer. No simulator."""

import importlib.util
import sys
import types

import gymnasium as gym
import pytest
import yaml

import moveitmoveit

if importlib.util.find_spec("isaaclab") is None:
    # the agents only import DirectRLEnv for type hints; stand it in so they import without Isaac Lab
    envs = types.ModuleType("isaaclab.envs")
    envs.DirectRLEnv = object
    sys.modules["isaaclab"] = types.ModuleType("isaaclab")
    sys.modules["isaaclab.envs"] = envs

from moveitmoveit.agents import AMP, PPO, AMPCfg, PPOCfg, resolve_agent  # noqa: E402

TASK_ID = "MoveIt-Humanoid-v0"


def agent_cfg(algo: str) -> dict:
    path = gym.spec(TASK_ID).kwargs[moveitmoveit.agent_cfg_entry_point(algo)]
    with open(path) as f:
        return yaml.safe_load(f)


@pytest.mark.parametrize(
    ("algo", "file", "agent_cls", "cfg_cls"),
    [("PPO", "ppo.yaml", PPO, PPOCfg), ("AMP", "amp.yaml", AMP, AMPCfg)],
)
def test_algo_loads_its_config_and_trainer(algo, file, agent_cls, cfg_cls):
    path = gym.spec(TASK_ID).kwargs[moveitmoveit.agent_cfg_entry_point(algo)]
    assert path == f"{moveitmoveit.CONFIGS_DIR}/agents/{file}"

    cfg = agent_cfg(algo)
    assert cfg["agent"]["class_type"] == algo

    # what OnPolicyRunner builds from it
    resolved_agent, resolved_cfg = resolve_agent(cfg["agent"]["class_type"])
    assert resolved_agent is agent_cls
    assert resolved_cfg is cfg_cls
    assert type(resolved_cfg(**cfg["agent"])) is cfg_cls


def test_ppo_config_has_no_discriminator():
    cfg = agent_cfg("PPO")
    assert "discriminator" not in cfg["models"]
    assert "style_reward_lambda" not in cfg["agent"]


def test_unknown_algo_is_rejected():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        moveitmoveit.agent_cfg_entry_point("SAC")
