"""Standing mode's rewards: per-env reward selection, the standing terms, the env glue and AMP's
style gate."""

import math
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from conftest import REPO_ROOT  # noqa: E402
from moveitmoveit.manifest import COMMAND_DIM, CommandIndex, load_manifest  # noqa: E402
from moveitmoveit.mdp import REWARDS, BodyState, ComposedMDP, StepContext  # noqa: E402
from moveitmoveit.motion_set import MotionSet  # noqa: E402

OBS = ("root_height",)
GOAL = slice(CommandIndex.GOAL_X, CommandIndex.GOAL_Y + 1)
STANDING_TERMS = ["root_lin_vel_l2", "root_ang_vel_l2", "upright", "support_centered"]


def still_state(n: int) -> BodyState:
    """Upright, motionless roots at 1 m with both feet (the last two key bodies) right below them."""
    return BodyState(
        torch.zeros(n, 28),
        torch.zeros(n, 28),
        torch.tensor([[0.0, 0.0, 1.0]]).repeat(n, 1),
        torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n, 4, 3),
    )


def context(state: BodyState, commands: torch.Tensor | None = None) -> StepContext:
    n = state.root_positions.shape[0]

    def no_reference():
        raise AssertionError("standing terms must not sample the reference motion")

    commands = torch.zeros(n, COMMAND_DIM) if commands is None else commands
    return StepContext(state, torch.zeros(n, 28), commands, torch.zeros(n, 28), torch.zeros(n, 28), no_reference)


@pytest.fixture
def specs(write_manifest):
    def _specs(motions: dict):
        return load_manifest(write_manifest({name: {"file": f"{name}.npz", **motion} for name, motion in motions.items()}))

    return _specs


# --- per-env reward selection -----------------------------------------------------------------


@pytest.fixture
def mdp(specs) -> ComposedMDP:
    return ComposedMDP(
        specs(
            {
                "walk": {
                    "rewards": {"lin_vel_tracking": 3.0, "joint_vel": {"weight": 1.0, "scale": 7.0}},
                    "standing": {"rewards": {"root_lin_vel_l2": 2.0, "joint_vel": {"weight": 0.5, "scale": 0.5}}},
                },
                "plain": {"rewards": {"lin_vel_tracking": 1.0, "joint_vel": 1.0}},
            }
        ),
        OBS,
        OBS,
        "cpu",
    )


def moving_context(n: int) -> StepContext:
    torch.manual_seed(0)
    state = still_state(n)._replace(
        dof_velocities=torch.randn(n, 28), root_linear_velocities=torch.randn(n, 3), root_rotations=torch.tensor([[0.0, 0.0, 0.0, 1.0]]).repeat(n, 1)
    )
    commands = torch.zeros(n, COMMAND_DIM)
    commands[:, CommandIndex.LIN_X] = torch.rand(n)
    return context(state, commands)


def test_standing_only_terms_join_the_selection_in_registry_order(mdp):
    assert mdp.reward_names == ["lin_vel_tracking", "joint_vel", "root_lin_vel_l2"]
    assert mdp.reward_weights.tolist() == [[3.0, 1.0, 0.0], [1.0, 1.0, 0.0]]
    assert mdp.standing_reward_weights.tolist() == [[0.0, 0.5, 2.0], [1.0, 1.0, 0.0]]  # plain keeps its own


def test_no_mask_and_an_all_false_mask_score_as_before(mdp):
    motion_ids = torch.tensor([0, 1, 0, 1])
    c = moving_context(4)

    before = mdp.rewards(c, motion_ids)
    masked = mdp.rewards(c, motion_ids, torch.zeros(4, dtype=torch.bool))

    for expected, actual in zip(before, masked):
        if isinstance(expected, dict):
            assert expected.keys() == actual.keys()
            assert all(torch.equal(expected[name], actual[name]) for name in expected)
        else:
            assert torch.equal(expected, actual)


def test_standing_envs_swap_to_standing_weights_and_kwargs_and_others_do_not_change(mdp):
    motion_ids = torch.tensor([0, 0, 1, 1])
    standing = torch.tensor([True, False, True, False])
    c = moving_context(4)
    s = c.state

    total, raw, weighted = mdp.rewards(c, motion_ids, standing)
    _, _, unmasked = mdp.rewards(c, motion_ids)

    dof_speed = torch.sum(s.dof_velocities**2, dim=-1)
    # env 0 stands: its joint_vel uses the standing scale and weight; walking terms weigh 0
    assert raw["joint_vel"][0] == pytest.approx(math.exp(-0.5 * dof_speed[0].item()))
    assert weighted["joint_vel"][0] == pytest.approx(0.5 * math.exp(-0.5 * dof_speed[0].item()))
    assert weighted["lin_vel_tracking"][0] == 0.0
    assert weighted["root_lin_vel_l2"][0] == pytest.approx(2.0 * math.exp(-4.0 * torch.sum(s.root_linear_velocities[0] ** 2).item()))
    # env 1 walks, and env 2's motion has no standing mode, so the flag changes nothing for it
    for env in (1, 2, 3):
        for name in unmasked:
            assert weighted[name][env] == unmasked[name][env], (env, name)
    assert raw["joint_vel"][1] == pytest.approx(math.exp(-7.0 * dof_speed[1].item()))
    assert torch.allclose(total, sum(weighted[name] for name in mdp.reward_names))


def test_unknown_standing_terms_are_rejected(specs):
    with pytest.raises(ValueError, match="standing.rewards"):
        ComposedMDP(specs({"m": {"standing": {"rewards": {"stand_stil": 1.0}}}}), OBS, OBS, "cpu")
    with pytest.raises(ValueError, match="standing.rewards.upright"):
        ComposedMDP(specs({"m": {"standing": {"rewards": {"upright": {"scael": 1.0}}}}}), OBS, OBS, "cpu")


@pytest.mark.parametrize("manifest", ["manifest.yaml", "goal_manifest.yaml"])
def test_shipped_manifests_compose_their_standing_rewards(manifest):
    joint_names = np.load(REPO_ROOT / "data/walk.npz")["dof_names"].tolist()  # the robot's joints
    mdp = ComposedMDP(load_manifest(str(REPO_ROOT / "configs" / manifest)), OBS, OBS, "cpu", joint_names=joint_names)

    assert set(STANDING_TERMS) <= set(mdp.reward_names)
    for name in STANDING_TERMS:
        column = mdp.reward_names.index(name)
        assert mdp.reward_weights[0, column] == 0.0 and mdp.standing_reward_weights[0, column] > 0.0


# --- standing terms ---------------------------------------------------------------------------


def test_a_still_upright_robot_over_its_feet_scores_one_on_every_standing_term():
    c = context(still_state(3))
    for name in STANDING_TERMS:
        value = REWARDS[name].fn(c, **{k: torch.full((3,), v) for k, v in REWARDS[name].defaults.items()})
        assert value.tolist() == pytest.approx([1.0] * 3), name


def test_standing_terms_fall_with_movement_tilt_and_an_off_center_root():
    half = math.sin(math.pi / 4)
    state = still_state(2)._replace(
        root_linear_velocities=torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        root_angular_velocities=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        root_rotations=torch.tensor([[0.0, half, 0.0, half], [0.0, 0.0, 0.0, 1.0]]),  # pitched 90 degrees
        key_body_positions=torch.tensor([[[0.0, 0.0, 0.0]] * 4, [[9.0, 9.0, 1.0]] * 2 + [[0.1, 0.0, 0.0]] * 2]),
    )
    c, one = context(state), torch.ones(2)

    assert REWARDS["root_lin_vel_l2"].fn(c, scale=one).tolist() == pytest.approx([math.exp(-0.25), 1.0])
    assert REWARDS["root_ang_vel_l2"].fn(c, scale=one).tolist() == pytest.approx([math.exp(-1.0), 1.0])
    assert REWARDS["upright"].fn(c, scale=one).tolist() == pytest.approx([math.exp(-1.0), 1.0], abs=1e-6)
    # the hands (first two key bodies) don't count; env 1's feet are 0.1 m off
    assert REWARDS["support_centered"].fn(c, scale=one).tolist() == pytest.approx([1.0, math.exp(-0.01)])


# --- env glue ---------------------------------------------------------------------------------


@pytest.fixture
def env_class(monkeypatch):
    """MotionLearningEnv's methods, importable without Isaac Sim: its base class and config are stubbed."""
    monkeypatch.setitem(sys.modules, "isaaclab.envs", types.SimpleNamespace(DirectRLEnv=object))
    monkeypatch.setitem(sys.modules, "moveitmoveit.env_cfg", types.SimpleNamespace(MotionLearningEnvCfg=object))
    monkeypatch.delitem(sys.modules, "moveitmoveit.env", raising=False)
    from moveitmoveit.env import MotionLearningEnv

    yield MotionLearningEnv
    sys.modules.pop("moveitmoveit.env", None)


@pytest.fixture
def env_motions(write_manifest, write_clip) -> MotionSet:
    return MotionSet.from_manifest(
        write_manifest(
            {
                "walk": {
                    "file": write_clip("a", 5),
                    "commands": {"lin_x": [0.5, 1.0]},
                    "zero_command_prob": 0.5,
                    "rewards": {"lin_vel_tracking": 3.0},
                    "standing": {"rewards": {"root_lin_vel_l2": 2.0}},
                },
                "to_goal": {
                    "file": write_clip("b", 6),
                    "commands": {"goal_x": [2.0, 3.0]},
                    "zero_command_prob": 0.5,
                    "rewards": {"target_hit": 1.0},
                    "standing": {"goal_radius": 0.3, "rewards": {"root_lin_vel_l2": 2.0}},
                },
                "plain": {"file": write_clip("c", 7), "commands": {"lin_x": [0.5, 1.0]}, "rewards": {"lin_vel_tracking": 3.0}},
            }
        ),
        "cpu",
    )


def make_env(env_class, motions: MotionSet, motion_ids: list[int], resample: bool = False):
    """The env's per-step state on stand-ins for Isaac Lab's: body 0 is the torso, 1-4 the key bodies."""
    n = len(motion_ids)
    env = object.__new__(env_class)
    env.cfg = SimpleNamespace(resample_commands=resample, command_resample_steps=150, random_reset=True, num_amp_observations=2)
    env.render_enabled = False
    env.device, env.num_envs, env.step_dt = "cpu", n, 1 / 30
    env.motions = motions
    env.mdp = ComposedMDP(motions.specs, OBS, OBS, "cpu")
    env.motion_ids = torch.tensor(motion_ids)
    env.commands = torch.zeros(n, COMMAND_DIM)
    env.standing = torch.zeros(n, dtype=torch.bool)
    env.episode_length_buf = torch.zeros(n, dtype=torch.long)
    env.extras = {}
    env.ref_body_index, env.key_body_indexes = 0, [1, 2, 3, 4]
    env.amp_observation_size = 2
    env.amp_observation_buffer = torch.zeros(n, 2, 1)
    env.actions = torch.zeros(n, 28)
    env.previous_actions = torch.zeros(n, 28)

    body_pos = torch.zeros(n, 5, 3)
    body_pos[:, 0, 2] = 1.0
    data = SimpleNamespace(
        joint_pos=torch.zeros(n, 28),
        joint_vel=torch.zeros(n, 28),
        joint_acc=torch.zeros(n, 28),
        body_pos_w=body_pos,
        body_quat_w=torch.tensor([0.0, 0.0, 0.0, 1.0]).repeat(n, 5, 1),
        body_lin_vel_w=torch.zeros(n, 5, 3),
        body_ang_vel_w=torch.zeros(n, 5, 3),
    )
    env.robot = SimpleNamespace(data=SimpleNamespace(**{name: SimpleNamespace(torch=value) for name, value in vars(data).items()}))
    return env


def test_env_observes_each_envs_mode_and_scores_it(env_class, env_motions):
    env = make_env(env_class, env_motions, [0, 0, 1, 1, 2])
    env.commands[1, CommandIndex.LIN_X] = 0.8
    env.commands[3, GOAL] = torch.tensor([2.5, 0.0])  # the goal motion's velocity slots stay 0
    env.commands[2, GOAL] = torch.tensor([0.0, 0.0])  # a zeroed goal: where the root is

    env._get_observations()
    assert env.standing.tolist() == [True, False, True, False, False]

    total = env._get_rewards()
    weighted = env.extras["reward_terms"]
    assert env.extras["standing"].tolist() == [True, False, True, False, False]
    assert weighted["root_lin_vel_l2"].tolist() == pytest.approx([2.0, 0.0, 2.0, 0.0, 0.0])  # still robot
    assert weighted["lin_vel_tracking"].tolist() == pytest.approx([0.0, 3.0 * math.exp(-2.0 * 0.64), 0.0, 0.0, 3.0])
    assert weighted["target_hit"].tolist() == pytest.approx([0.0, 0.0, 0.0, math.exp(-0.5 * 6.25), 0.0])
    assert torch.allclose(total, weighted["task"])


def test_resampled_commands_switch_modes_and_rewards_follow_the_observed_mode(env_class, env_motions):
    torch.manual_seed(0)
    env = make_env(env_class, env_motions, [0] * 32 + [1] * 32, resample=True)

    def zero_command():
        offset = env.commands[:, GOAL] - env.robot.data.body_pos_w.torch[:, 0, :2]
        return torch.cat((env.commands[:32, : CommandIndex.YAW + 1].abs().sum(-1) == 0, torch.all(offset[32:] == 0, dim=-1)))

    env._get_observations()  # episode step 0: every env samples a command
    first = env.standing.clone()
    assert torch.equal(first, zero_command())

    env._get_observations()  # and samples again: half the commands are zeroed, so modes switch
    second = env.standing.clone()
    assert torch.equal(second, zero_command())
    assert (first & ~second).any() and (~first & second).any()

    # the step after: its reward is for the action the policy chose seeing `second`
    env.episode_length_buf += 1
    env._get_rewards()
    assert torch.equal(env.extras["standing"], second)
    assert torch.equal(env.extras["reward_terms"]["root_lin_vel_l2"] > 0, second)


def test_reward_uses_the_mode_the_policy_acted_on_until_the_next_observation(env_class, env_motions):
    env = make_env(env_class, env_motions, [0])
    env._get_observations()
    assert env.standing.tolist() == [True]

    env.commands[0, CommandIndex.LIN_X] = 0.8  # e.g. play's keyboard writing before env.step
    env._get_rewards()
    assert env.extras["standing"].tolist() == [True]
    standing_mask = env.extras["standing"]

    env._get_observations()
    env._get_rewards()
    assert env.extras["standing"].tolist() == [False]
    assert standing_mask.tolist() == [True]  # the mask AMP got for the earlier step isn't overwritten
    assert env.extras["reward_terms"]["lin_vel_tracking"][0] > 0 and env.extras["reward_terms"]["root_lin_vel_l2"][0] == 0


def test_a_goal_env_stands_on_arrival_and_walks_again_when_it_drifts_off(env_class, env_motions):
    env = make_env(env_class, env_motions, [1])
    env.commands[0, GOAL] = torch.tensor([10.0, 5.0])
    root = env.robot.data.body_pos_w.torch[:, 0]
    modes = []
    for xy, yaw in [((9.0, 5.0), 0.0), ((9.8, 5.1), 0.0), ((9.8, 5.1), 2.0), ((10.0, 5.4), 0.0)]:
        root[0, :2] = torch.tensor(xy)
        env.robot.data.body_quat_w.torch[0, 0] = torch.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])
        env._get_observations()
        modes.append(env.standing.item())

    # 1 m out walks; 0.22 m out stands, whichever way it faces; 0.4 m out walks again
    assert modes == [False, True, True, False]


# --- AMP style gate ---------------------------------------------------------------------------


@pytest.fixture
def amp(monkeypatch):
    """An AMP agent with only what process_env_step reads; PPO's step is captured, not run."""
    # the agents only name DirectRLEnv in type hints
    monkeypatch.setitem(sys.modules, "isaaclab.envs", types.SimpleNamespace(DirectRLEnv=object))
    for name in [name for name in sys.modules if name.startswith("moveitmoveit.agents")]:
        monkeypatch.delitem(sys.modules, name)
    from moveitmoveit.agents.amp import AMP, AMPCfg
    from moveitmoveit.agents.ppo import PPO

    captured, appended = {}, []

    def capture(self, next_observations, rewards, terminated, truncated, infos=None):
        captured.update(rewards=rewards, infos=infos)

    monkeypatch.setattr(PPO, "process_env_step", capture)
    agent = object.__new__(AMP)
    agent.cfg = AMPCfg(class_type="AMP", style_reward_lambda=2.0, goal_reward_lambda=0.5)
    logits = torch.tensor([[0.0], [1.0], [-1.0], [2.0]])
    agent.discriminator = lambda amp_obs: logits
    agent._disc_obs_preprocessor = lambda amp_obs: amp_obs
    agent._amp_observations_buf = SimpleNamespace(append=appended.append)
    style = -torch.log(1 - torch.sigmoid(logits)).view(-1)

    def step(infos):
        task = torch.tensor([1.0, 2.0, 3.0, 4.0])
        amp_obs = torch.arange(8.0).view(4, 2)
        agent.process_env_step(torch.zeros(4, 3), task, torch.zeros(4, dtype=torch.bool), torch.zeros(4, dtype=torch.bool), {"amp_obs": amp_obs, "reward_terms_raw": {}, "reward_terms": {}, **infos})
        return task, amp_obs

    yield SimpleNamespace(step=step, captured=captured, appended=appended, style=style)
    for name in [name for name in sys.modules if name.startswith("moveitmoveit.agents")]:
        sys.modules.pop(name)


def test_style_reward_is_off_only_for_flagged_envs_and_the_discriminator_still_sees_them(amp):
    task, amp_obs = amp.step({"style_reward_off": torch.tensor([True, False, True, False])})

    on = torch.tensor([False, True, False, True])
    expected = 0.5 * task + torch.where(on, 2.0 * amp.style, torch.zeros(4))
    assert torch.allclose(amp.captured["rewards"], expected)  # what PPO stores in the rollout
    assert torch.allclose(amp.captured["infos"]["reward_terms"]["style"], torch.where(on, 2.0 * amp.style, torch.zeros(4)))
    assert torch.allclose(amp.captured["infos"]["reward_terms_raw"]["style"], amp.style)  # raw stays the discriminator's
    assert len(amp.appended) == 1 and torch.equal(amp.appended[0], amp_obs)  # every env's AMP obs, as before


@pytest.mark.parametrize("infos", [{}, {"style_reward_off": torch.zeros(4, dtype=torch.bool)}], ids=["missing", "all_false"])
def test_without_a_flagged_env_every_env_earns_style_as_before(amp, infos):
    task, _ = amp.step(infos)
    assert torch.allclose(amp.captured["rewards"], 0.5 * task + 2.0 * amp.style)


def test_every_env_flagged_leaves_only_the_task_reward(amp):
    task, _ = amp.step({"style_reward_off": torch.ones(4, dtype=torch.bool)})
    assert torch.allclose(amp.captured["rewards"], 0.5 * task)
    assert len(amp.appended) == 1  # the discriminator still sees them


def test_the_standing_mask_alone_no_longer_gates_style(amp):
    # AMP reads only `style_reward_off`; the env derives it from standing (see the env tests below)
    task, _ = amp.step({"standing": torch.ones(4, dtype=torch.bool)})
    assert torch.allclose(amp.captured["rewards"], 0.5 * task + 2.0 * amp.style)


def test_the_env_turns_style_off_exactly_for_standing_envs(env_class, env_motions, amp):
    env = make_env(env_class, env_motions, [0, 0, 1, 1])
    env.commands[1, CommandIndex.LIN_X] = 0.8
    env.commands[3, GOAL] = torch.tensor([2.5, 0.0])
    env._get_observations()
    env._get_rewards()
    assert env.extras["style_reward_off"].tolist() == env.extras["standing"].tolist() == [True, False, True, False]

    task, _ = amp.step({"style_reward_off": env.extras["style_reward_off"]})
    assert torch.allclose(amp.captured["rewards"], 0.5 * task + torch.tensor([0.0, 1.0, 0.0, 1.0]) * 2.0 * amp.style)


def test_another_condition_can_turn_style_off_without_putting_envs_in_standing_mode(env_class, env_motions):
    class FlaggedEnv(env_class):
        def _style_reward_off(self):
            return super()._style_reward_off() | self.flagged

    env = make_env(FlaggedEnv, env_motions, [0, 0, 2])
    env.flagged = torch.tensor([False, True, True])
    env.commands[1:, CommandIndex.LIN_X] = 0.8  # envs 1 and 2 walk
    env._get_observations()
    env._get_rewards()

    assert env.extras["standing"].tolist() == [True, False, False]
    assert env.extras["style_reward_off"].tolist() == [True, True, True]
    weighted = env.extras["reward_terms"]
    assert weighted["root_lin_vel_l2"].tolist() == pytest.approx([2.0, 0.0, 0.0])  # standing rewards only for env 0
    assert weighted["lin_vel_tracking"][1:].tolist() == pytest.approx([3.0 * math.exp(-2.0 * 0.64)] * 2)  # flagged envs keep theirs
