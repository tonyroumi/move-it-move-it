import gymnasium as gym
import pytest
import yaml

import moveitmoveit


@pytest.mark.parametrize("algo", sorted(moveitmoveit.ALGO_CFG_ENTRY_POINTS))
def test_algo_loads_its_own_agent_config(algo):
    entry_point = moveitmoveit.ALGO_CFG_ENTRY_POINTS[algo]
    cfg_path = gym.spec("MoveIt-Humanoid-v0").kwargs[entry_point]

    with open(cfg_path) as f:
        agent_cfg = yaml.safe_load(f)

    assert agent_cfg["agent"]["class_type"] == algo
