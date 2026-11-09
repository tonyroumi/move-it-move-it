"""Key-triggered behaviors: the latching state machine, what the policy observes, the manifest
fields, the env's use of them in training and play, the heading-frame tracking reward, and the
keyboard. No simulator is started.

Run under IsaacLab's Python, which has torch: ./isaaclab.sh -p -m pytest tests
"""

import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

pytest.importorskip("isaaclab.utils.math", reason="needs IsaacLab's quaternion math")

from moveitmoveit.behavior import (  # noqa: E402
    LOCOMOTION,
    BehaviorManager,
    RandomKeyPresses,
    behavior_commands,
    settled,
)
from moveitmoveit.manifest import COMMAND_DIM, CommandIndex, SettleSpec, load_manifest  # noqa: E402
from moveitmoveit.mdp import REWARDS, BodyState, StepContext  # noqa: E402
from moveitmoveit.motion.reference import TrackingTargets  # noqa: E402
from moveitmoveit.motion_set import MotionSet  # noqa: E402
from moveitmoveit.utils import transforms  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
BOXING_MANIFEST = REPO_ROOT / "configs/boxing_manifest.yaml"
DT = 0.125  # exact in binary, so phases land exactly on their thresholds


@pytest.fixture
def write_manifest(tmp_path):
    def _write(motions: dict) -> str:
        path = tmp_path / "manifest.yaml"
        path.write_text(yaml.safe_dump({"motions": motions}, sort_keys=False))
        return str(path)

    return _write


@pytest.fixture
def write_clip(tmp_path):
    """A synthetic 60 fps clip of `num_frames` frames, whose fields ramp with the frame index."""

    def _write(name: str, num_frames: int) -> str:
        ramp = np.arange(num_frames, dtype=np.float32)
        rotations = np.zeros((num_frames, 3, 4), dtype=np.float32)
        rotations[..., 2] = np.sin(0.01 * ramp)[:, None]
        rotations[..., 3] = np.cos(0.01 * ramp)[:, None]
        body = ramp[:, None, None] * np.arange(1, 10, dtype=np.float32).reshape(3, 3)
        path = tmp_path / f"{name}.npz"
        np.savez(
            path,
            fps=np.array(60.0),
            dof_names=np.array(["dof_a", "dof_b", "dof_c"]),
            body_names=np.array(["torso", "hand", "foot"]),
            dof_positions=ramp[:, None] * np.arange(1, 4, dtype=np.float32),
            dof_velocities=2.0 * ramp[:, None] * np.arange(1, 4, dtype=np.float32),
            body_positions=body,
            body_rotations=rotations,
            body_linear_velocities=3.0 * body,
            body_angular_velocities=4.0 * body,
        )
        return str(path)

    return _write


class Driver:
    """Steps a BehaviorManager the way the env does, with a fixed behavior duration."""

    def __init__(self, num_envs: int = 1, num_behaviors: int = 2, duration: float = 0.5):
        self.manager = BehaviorManager(num_envs, num_behaviors, "cpu")
        self.duration = torch.full((num_envs,), duration)
        self.time = torch.zeros(num_envs)

    def step(self, *keys: float) -> tuple[int, float]:
        """Advance one step with these key levels; the env's mode and phase afterwards."""
        self.time += DT
        complete = self.manager.phase(self.time, self.duration) >= 1.0
        self.manager.update(torch.tensor([keys]), complete, self.time)
        return int(self.manager.mode[0]), float(self.manager.phase(self.time, self.duration)[0])


# --- the state machine ------------------------------------------------------------------------


def test_a_press_starts_the_behavior_and_release_does_not_cancel_it():
    driver = Driver()
    assert driver.step(0, 0) == (LOCOMOTION, 0.0)

    assert driver.step(0, 1) == (1, 0.0)  # press behavior 1
    assert driver.step(0, 0) == (1, 0.25)  # release at once: it keeps going
    assert [driver.step(0, 0) for _ in range(2)] == [(1, 0.5), (1, 0.75)]

    # phase min(1, (t - t_start) / duration) reaches 1: that step is back in locomotion
    assert driver.step(0, 0) == (LOCOMOTION, 0.0)


def test_holding_a_key_through_the_end_does_not_start_it_again():
    driver = Driver()
    driver.step(1, 0)
    modes = [driver.step(1, 0)[0] for _ in range(8)]
    assert modes == [0] * 3 + [LOCOMOTION] * 5

    driver.step(0, 0)  # only a fresh press starts it
    assert driver.step(1, 0) == (0, 0.0)


def test_presses_while_a_behavior_runs_are_ignored():
    driver = Driver(duration=1.0)
    driver.step(1, 0)
    driver.step(0, 0)
    assert driver.step(0, 1) == (0, 0.25)  # another behavior's key
    driver.step(0, 0)
    assert driver.step(1, 0) == (0, 0.5)  # its own key again: no restart


def test_the_lowest_behavior_wins_a_tie():
    driver = Driver(num_behaviors=3)
    assert driver.step(0, 1, 1)[0] == 1


def test_a_press_on_the_step_a_behavior_completes_starts_the_next():
    driver = Driver(duration=0.25)
    driver.step(1, 0)
    driver.step(0, 0)
    assert driver.step(0, 1) == (1, 0.0)  # phase 1 and a press of behavior 1, in one step


def test_reset_latches_a_behavior_and_swallows_a_key_held_across_it():
    manager = BehaviorManager(2, 2, "cpu")
    duration = torch.tensor([1.0, 1.0])
    # env 0 resets 0.3 s into behavior 1, env 1 into locomotion with key 0 held
    manager.reset(torch.tensor([0, 1]), torch.tensor([[0.0, 0.0], [1.0, 0.0]]), torch.tensor([1, LOCOMOTION]), torch.tensor([-0.3, 0.0]))

    time = torch.zeros(2)
    assert manager.phase(time, duration).tolist() == pytest.approx([0.3, 0.0])
    manager.update(torch.tensor([[0.0, 0.0], [1.0, 0.0]]), torch.zeros(2, dtype=torch.bool), time)
    assert manager.mode.tolist() == [1, LOCOMOTION]


def test_no_behaviors_is_a_no_op():
    manager = BehaviorManager(3, 0, "cpu")
    started = manager.update(torch.zeros(3, 0), torch.ones(3, dtype=torch.bool), torch.ones(3))
    assert not started.any() and manager.mode.tolist() == [LOCOMOTION] * 3


# --- stable posture ---------------------------------------------------------------------------


def test_a_still_robot_settles_only_past_min_phase():
    still, moving = torch.zeros(3, 28), torch.full((3, 28), 2.0)
    phase = torch.tensor([0.0, 0.79, 0.8])
    min_phase, max_speed = torch.full((3,), 0.8), torch.full((3,), 0.6)

    # still in its guard before the move (phase 0) does not count
    assert settled(phase, still, min_phase, max_speed).tolist() == [False, False, True]
    assert not settled(phase, moving, min_phase, max_speed).any()
    assert not settled(torch.ones(3), still, torch.full((3,), math.inf), max_speed).any()


# --- the policy's view ------------------------------------------------------------------------


def test_the_policy_sees_the_latched_behavior_and_its_phase_not_the_key():
    commands = torch.zeros(3, COMMAND_DIM)
    commands[:, CommandIndex.BINARY_KEY_0] = torch.tensor([0.0, 1.0, 1.0])  # raw levels
    commands[:, CommandIndex.BINARY_KEY_1] = torch.tensor([1.0, 0.0, 0.0])  # no behavior uses it
    commands[:, CommandIndex.GOAL_X] = 4.0
    keys = [int(CommandIndex.BINARY_KEY_2), int(CommandIndex.BINARY_KEY_0)]

    observed = behavior_commands(commands, keys, torch.tensor([0, LOCOMOTION, 1]), torch.tensor([0.25, 0.0, 0.5]))

    assert observed.shape == (3, COMMAND_DIM)
    assert observed[:, keys].tolist() == [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]]  # z(t)
    assert observed[:, CommandIndex.BEHAVIOR_PHASE].tolist() == [0.25, 0.0, 0.5]  # phi(t)
    assert torch.equal(observed[:, CommandIndex.BINARY_KEY_1], commands[:, CommandIndex.BINARY_KEY_1])
    assert (observed[:, CommandIndex.GOAL_X] == 4.0).all()  # the locomotion command passes through


def test_without_behaviors_only_the_phase_slot_changes():
    commands = torch.randn(2, COMMAND_DIM)
    observed = behavior_commands(commands, [], torch.full((2,), LOCOMOTION), torch.zeros(2))
    assert torch.equal(observed[:, : CommandIndex.BEHAVIOR_PHASE], commands[:, : CommandIndex.BEHAVIOR_PHASE])
    assert (observed[:, CommandIndex.BEHAVIOR_PHASE] == 0).all()


# --- training's key presses -------------------------------------------------------------------


def test_random_presses_tap_and_hold_with_a_release_between():
    torch.manual_seed(0)
    presses = RandomKeyPresses(64, rate=5.0, weights=torch.tensor([0.25, 0.75]), max_hold_steps=8, step_dt=DT, device="cpu")
    levels = torch.stack([presses.step() for _ in range(400)], dim=1)  # (env, step, behavior)

    assert ((levels == 0) | (levels == 1)).all() and (levels.sum(-1) <= 1).all()  # one key at a time
    held = levels.sum(-1) > 0
    lengths = []
    for env in range(64):
        run = 0
        for step in range(held.shape[1]):
            if held[env, step]:
                run += 1
                # a new press never follows another key without a release step
                if step > 0 and held[env, step - 1]:
                    assert torch.equal(levels[env, step], levels[env, step - 1])
            elif run:
                lengths.append(run)
                run = 0
    assert min(lengths) == 1 and max(lengths) == 8
    assert levels[..., 1].sum() > 2 * levels[..., 0].sum()


def test_random_presses_drive_the_same_latching_as_play():
    torch.manual_seed(1)
    num_envs, duration = 32, torch.full((32,), 0.5)
    presses = RandomKeyPresses(num_envs, rate=4.0, weights=torch.tensor([0.5, 0.5]), max_hold_steps=12, step_dt=DT, device="cpu")
    manager = BehaviorManager(num_envs, 2, "cpu")
    time, previous_levels, previous_mode = torch.zeros(num_envs), torch.zeros(num_envs, 2), manager.mode.clone()

    starts = 0
    for _ in range(500):
        time += DT
        levels = presses.step()
        complete = manager.phase(time, duration) >= 1.0
        started = manager.update(levels, complete, time)
        mode = manager.mode

        # a behavior starts only from locomotion, on a rising edge of its own key
        rising = (levels > 0.5) & (previous_levels <= 0.5)
        assert ((previous_mode[started] == LOCOMOTION) | complete[started]).all()
        assert rising[started, mode[started]].all()
        # it ends only once complete, whatever the keys do meanwhile
        ended = (previous_mode != LOCOMOTION) & (mode != previous_mode)
        assert complete[ended].all()
        starts += int(started.sum())
        previous_levels, previous_mode = levels, mode.clone()
    assert starts > 100


# --- the manifest -------------------------------------------------------------------------------


def test_triggered_behaviors_parse(write_manifest):
    specs = load_manifest(
        write_manifest(
            {
                "walk": {"file": "walk.npz", "commands": {"goal_x": [-1.0, 1.0]}},
                "jab": {"file": "jab.npz", "trigger": "binary_key_2", "settle": {"min_phase": 0.8, "max_dof_speed": 0.6}},
            }
        )
    )
    assert (specs[0].trigger, specs[0].settle) == (None, None)
    assert specs[1].trigger == "binary_key_2"
    assert specs[1].settle == SettleSpec(0.8, 0.6)


@pytest.mark.parametrize(
    "jab, message",
    [
        ({"trigger": "lin_x"}, "trigger"),
        ({"trigger": ["binary_key_0"]}, "trigger"),
        ({"trigger": "binary_key_0", "commands": {"lin_x": 1.0}}, "declare none"),
        ({"trigger": "binary_key_0", "sample_prob": 0}, "positive sample_prob"),
        ({"settle": {"min_phase": 0.5, "max_dof_speed": 1.0}}, "only a triggered behavior"),
        ({"trigger": "binary_key_0", "settle": {"min_phase": 0.0, "max_dof_speed": 1.0}}, "min_phase"),
        ({"trigger": "binary_key_0", "settle": {"min_phase": 0.5, "max_dof_speed": 0.0}}, "max_dof_speed"),
        ({"trigger": "binary_key_0", "settle": {"min_phase": 0.5}}, "missing"),
        ({"trigger": "binary_key_0", "settle": {"min_phase": 0.5, "max_dof_speed": 1.0, "x": 1}}, "unknown"),
        ({"commands": {"behavior_phase": 0.5}}, "behavior_phase"),
        ({"trigger": "binary_key_0", "standing": {}}, "no standing mode"),
    ],
)
def test_invalid_behaviors_are_rejected(write_manifest, jab, message):
    with pytest.raises(ValueError, match=message):
        load_manifest(write_manifest({"walk": {"file": "walk.npz"}, "jab": {"file": "jab.npz", **jab}}))


def test_trigger_slots_are_exclusive(write_manifest):
    with pytest.raises(ValueError, match="more than one"):
        load_manifest(
            write_manifest(
                {
                    "walk": {"file": "walk.npz"},
                    "jab": {"file": "jab.npz", "trigger": "binary_key_0"},
                    "hook": {"file": "hook.npz", "trigger": "binary_key_0"},
                }
            )
        )
    with pytest.raises(ValueError, match="no motion may declare"):
        load_manifest(
            write_manifest(
                {"walk": {"file": "walk.npz", "commands": {"binary_key_0": 1.0}}, "jab": {"file": "jab.npz", "trigger": "binary_key_0"}}
            )
        )


def test_behaviors_need_a_locomotion_motion_to_return_to(write_manifest):
    with pytest.raises(ValueError, match="locomotion motion"):
        load_manifest(
            write_manifest({"walk": {"file": "walk.npz", "sample_prob": 0}, "jab": {"file": "jab.npz", "trigger": "binary_key_0"}})
        )


def test_behaviors_and_a_conditional_discriminator_are_not_combined(write_manifest):
    # the running behavior's one-hot shares the binary key slots the discriminator's labels come from
    with pytest.raises(ValueError, match="style_condition"):
        load_manifest(
            write_manifest(
                {
                    "walk": {"file": "walk.npz", "commands": {"lin_x": 1.0}, "style_condition": 0},
                    "jab": {"file": "jab.npz", "trigger": "binary_key_1", "style_condition": 0},
                }
            )
        )


def test_shipped_boxing_manifest_is_valid():
    specs = load_manifest(str(BOXING_MANIFEST))
    assert {spec.name: spec.trigger for spec in specs if spec.trigger} == {
        "jab": "binary_key_0",
        "right_hook": "binary_key_1",
        "combo": "binary_key_2",
    }


# --- motion tables ----------------------------------------------------------------------------


@pytest.fixture
def boxing(write_manifest, write_clip) -> MotionSet:
    """Listed out of key order on purpose: hook's slot 0 makes it behavior 0."""
    return MotionSet.from_manifest(
        write_manifest(
            {
                "walk": {"file": write_clip("walk", 11), "sample_prob": 0.5, "commands": {"goal_x": [1.0, 2.0]}},
                "jab": {"file": write_clip("jab", 7), "trigger": "binary_key_1", "sample_prob": 0.3},
                "hook": {
                    "file": write_clip("hook", 13),
                    "trigger": "binary_key_0",
                    "sample_prob": 0.2,
                    "settle": {"min_phase": 0.5, "max_dof_speed": 0.6},
                },
            }
        ),
        "cpu",
    )


def test_behaviors_are_numbered_by_key_slot_not_manifest_order(boxing):
    assert boxing.num_behaviors == 2
    assert boxing.behavior_keys == [CommandIndex.BINARY_KEY_0, CommandIndex.BINARY_KEY_1]
    assert boxing.behavior_motion_ids.tolist() == [boxing.name_to_id["hook"], boxing.name_to_id["jab"]]
    assert boxing.motion_behavior.tolist() == [LOCOMOTION, 1, 0]
    assert boxing.locomotion_sample_prob.tolist() == pytest.approx([1.0, 0.0, 0.0])
    assert boxing.settle_min_phase.tolist() == [math.inf, math.inf, 0.5]


def test_trigger_rate_matches_the_reference_share_of_each_behavior(boxing):
    p = boxing.sample_prob[boxing.behavior_motion_ids]
    d = boxing.library.duration[boxing.behavior_motion_ids]
    rate, q = boxing.trigger_rate, boxing.trigger_weights

    # share of agent time in behavior b: rate * q_b * d_b / (1 + rate * sum q d)
    share = rate * q * d / (1.0 + rate * (q * d).sum())
    assert share.tolist() == pytest.approx(p.tolist())


def test_no_behaviors_means_no_presses(write_manifest, write_clip):
    motions = MotionSet.from_manifest(write_manifest({"walk": {"file": write_clip("walk", 5)}}), "cpu")
    assert motions.num_behaviors == 0 and motions.trigger_rate == 0.0 and motions.behavior_keys == []


def test_shipped_boxing_manifest_loads_its_clips(monkeypatch):
    # data/ isn't tracked: the clips are only there once they have been put in place
    files = [REPO_ROOT / spec.file for spec in load_manifest(str(BOXING_MANIFEST))]
    if not all(file.exists() for file in files):
        pytest.skip("the boxing clips are not under data/boxing/")
    monkeypatch.chdir(REPO_ROOT)  # the manifest's clip paths are relative to the repo root
    motions = MotionSet.from_manifest(str(BOXING_MANIFEST), "cpu")
    names = [motions.names[i] for i in motions.behavior_motion_ids.tolist()]
    assert names == ["jab", "right_hook", "combo"]
    assert motions.has_goal[motions.motion_behavior < 0].all()
    assert not motions.has_goal[motions.behavior_motion_ids].any()
    assert 0.0 < motions.trigger_rate < 2.0


# --- the env's use of them ----------------------------------------------------------------------


@pytest.fixture
def env_class(monkeypatch):
    """MotionLearningEnv's methods, importable without Isaac Sim: its base class and config are stubbed."""
    monkeypatch.setitem(sys.modules, "isaaclab.envs", types.SimpleNamespace(DirectRLEnv=object))
    monkeypatch.setitem(sys.modules, "moveitmoveit.env_cfg", types.SimpleNamespace(MotionLearningEnvCfg=object))
    monkeypatch.delitem(sys.modules, "moveitmoveit.env", raising=False)
    from moveitmoveit.env import MotionLearningEnv

    yield MotionLearningEnv
    sys.modules.pop("moveitmoveit.env", None)


def fake_env(motions: MotionSet, num_envs: int, resample_commands: bool, step_dt: float = 1 / 30) -> SimpleNamespace:
    return SimpleNamespace(
        cfg=SimpleNamespace(resample_commands=resample_commands),
        motions=motions,
        num_envs=num_envs,
        step_dt=step_dt,
        episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
        commands=torch.zeros(num_envs, COMMAND_DIM),
        motion_ids=torch.zeros(num_envs, dtype=torch.long),
        base_motion_ids=torch.zeros(num_envs, dtype=torch.long),
        motion_start_times=torch.zeros(num_envs),
        behaviors=BehaviorManager(num_envs, motions.num_behaviors, "cpu"),
        behavior_phase=torch.zeros(num_envs),
        key_presses=RandomKeyPresses(num_envs, motions.trigger_rate, motions.trigger_weights, 30, step_dt, "cpu"),
        robot=SimpleNamespace(data=SimpleNamespace(joint_vel=SimpleNamespace(torch=torch.ones(num_envs, 28)))),
    )


def test_a_key_press_in_play_switches_the_env_to_the_behavior_until_its_clip_ends(env_class, boxing):
    env = fake_env(boxing, 1, resample_commands=False, step_dt=0.04)
    walk, jab = boxing.name_to_id["walk"], boxing.name_to_id["jab"]
    assert float(boxing.library.duration[jab]) == pytest.approx(0.1)
    observed = []

    def step(jab_key: float):
        env.episode_length_buf += 1
        env.commands[0, CommandIndex.BINARY_KEY_1] = jab_key  # as play's keyboard writes it
        env_class._update_behaviors(env)
        observed.append(behavior_commands(env.commands, boxing.behavior_keys, env.behaviors.mode, env.behavior_phase)[0])
        return int(env.motion_ids[0]), round(float(env.behavior_phase[0]), 6)

    assert step(0.0) == (walk, 0.0)
    assert step(1.0) == (jab, 0.0)
    assert step(0.0) == (jab, pytest.approx(0.4))  # released: keeps going
    assert step(1.0) == (jab, pytest.approx(0.8))  # pressed again: ignored
    assert step(1.0) == (walk, 0.0)  # phase reached 1; the held key does not restart it
    assert step(0.0) == (walk, 0.0)

    z = [row[boxing.behavior_keys].tolist() for row in observed]
    assert z == [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]


def test_a_running_behavior_is_never_in_standing_mode(env_class, write_manifest, write_clip):
    motions = MotionSet.from_manifest(
        write_manifest(
            {
                "walk": {"file": write_clip("walk", 11), "commands": {"lin_x": [0.5, 1.0]}, "standing": {"rewards": {"root_lin_vel_l2": 1.0}}},
                "jab": {"file": write_clip("jab", 7), "trigger": "binary_key_0"},
            }
        ),
        "cpu",
    )
    env = fake_env(motions, 2, resample_commands=False)  # zero commands: both would stand
    env.episode_length_buf += 1
    env.commands[0, CommandIndex.BINARY_KEY_0] = 1.0  # env 0 throws the jab
    env_class._update_behaviors(env)
    commands = behavior_commands(env.commands, motions.behavior_keys, env.behaviors.mode, env.behavior_phase)
    # so its style reward stays on (style_reward_off is the standing mask) and its own rewards apply
    assert motions.standing(env.motion_ids, commands).tolist() == [False, True]


def test_a_still_robot_ends_a_settling_behavior_early_but_not_before_min_phase(env_class, boxing):
    env = fake_env(boxing, 1, resample_commands=False, step_dt=0.03)
    env.robot.data.joint_vel.torch = torch.zeros(1, 28)  # still throughout, guard included
    assert float(boxing.library.duration[boxing.name_to_id["hook"]]) == pytest.approx(0.2)  # min_phase 0.5

    phases = []
    for step in range(12):
        env.episode_length_buf += 1
        env.commands[0, CommandIndex.BINARY_KEY_0] = 1.0 if step == 0 else 0.0
        env_class._update_behaviors(env)
        phases.append((int(env.behaviors.mode[0]), round(float(env.behavior_phase[0]), 3)))

    # pressed at step 0, so step k is at phase 0.15 k: past min_phase at k = 4, short of phase 1
    assert [mode for mode, _ in phases[:4]] == [0] * 4
    assert phases[3][1] == pytest.approx(0.45)
    assert phases[4] == (LOCOMOTION, 0.0)


def test_training_presses_write_raw_keys_that_the_same_latching_reads(env_class, boxing):
    torch.manual_seed(0)
    env = fake_env(boxing, 256, resample_commands=True)
    previous_mode = env.behaviors.mode.clone()
    previous_keys = env.commands[:, boxing.behavior_keys].clone()
    behavior_steps = 0
    for _ in range(300):
        env.episode_length_buf += 1
        env_class._update_behaviors(env)
        keys = env.commands[:, boxing.behavior_keys]
        mode = env.behaviors.mode
        started = (previous_mode == LOCOMOTION) & (mode != LOCOMOTION)
        assert ((keys > 0.5) & (previous_keys <= 0.5))[started, mode[started]].all()
        assert torch.equal(env.motion_ids[mode != LOCOMOTION], boxing.behavior_motion_ids[mode[mode != LOCOMOTION]])
        assert torch.equal(env.motion_ids[mode == LOCOMOTION], env.base_motion_ids[mode == LOCOMOTION])
        behavior_steps += int((mode != LOCOMOTION).sum())
        previous_mode, previous_keys = mode.clone(), keys.clone()
    assert behavior_steps > 0


def test_a_reset_into_a_behavior_clip_runs_it_from_there_and_returns_to_locomotion(env_class, boxing):
    env = fake_env(boxing, 2, resample_commands=True)
    jab, walk = boxing.name_to_id["jab"], boxing.name_to_id["walk"]
    env.motion_ids[:] = torch.tensor([jab, walk])
    env.commands[:, boxing.behavior_keys] = 1.0  # keys left held from before the reset
    times = torch.tensor([0.05, 0.1])
    env.motion_start_times[:] = times

    env_class._reset_behaviors(env, torch.tensor([0, 1]), times)
    env_class._update_behaviors(env)  # the reset's first observation, at episode time 0

    assert env.behaviors.mode.tolist() == [1, LOCOMOTION]
    assert env.base_motion_ids.tolist() == [walk, walk]
    assert env.behavior_phase[0] == pytest.approx(0.05 / float(boxing.library.duration[jab]))
    assert env.motion_start_times[1] == pytest.approx(0.1)  # locomotion keeps its clip time


def test_tracking_follows_the_behavior_clip_from_its_start(env_class, boxing):
    env = fake_env(boxing, 2, resample_commands=False)
    env.episode_length_buf[:] = 30  # 1 s into the episode
    env.motion_start_times[:] = torch.tensor([0.2, 0.2])
    env.behaviors.mode[:] = torch.tensor([0, LOCOMOTION])
    env.behaviors.start_time[:] = torch.tensor([0.75, 0.0])
    env.random_yaw_quat = None
    env.reference = SimpleNamespace(tracking_targets=lambda ids, times, yaw: times)

    assert env_class._current_tracking_targets(env).tolist() == pytest.approx([0.25, 1.2])


# --- heading-frame tracking -------------------------------------------------------------------


def yaw_quat(yaw: float) -> torch.Tensor:
    return torch.tensor([[0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])


def test_heading_tracking_scores_a_motion_the_same_at_any_heading_and_place():
    torch.manual_seed(0)
    root, keys = torch.tensor([[0.0, 0.0, 1.0]]), torch.randn(1, 4, 3)
    ref_root, ref_keys = torch.tensor([[0.0, 0.0, 1.0]]), keys + 0.05 * torch.randn(1, 4, 3)
    dofs, ref_dofs = torch.randn(1, 28), torch.randn(1, 28) * 0.01

    def score(robot_yaw: float, robot_shift: torch.Tensor, name: str) -> float:
        q = yaw_quat(robot_yaw)
        moved_keys = transforms.quat_apply(q.expand(4, 4), keys[0]).unsqueeze(0) + robot_shift
        state = BodyState(dofs, dofs, root + robot_shift, q, torch.zeros(1, 3), torch.zeros(1, 3), moved_keys)
        targets = TrackingTargets(ref_dofs, ref_root, ref_keys, yaw_quat(0.0))
        c = StepContext(state, torch.zeros(1, 28), torch.zeros(1, COMMAND_DIM), dofs, dofs, lambda: targets)
        term = REWARDS[name]
        return float(term.fn(c, **{k: torch.tensor(v) for k, v in term.defaults.items()}))

    base = score(0.0, torch.zeros(3), "heading_motion_tracking")
    assert score(2.0, torch.tensor([3.0, -4.0, 0.0]), "heading_motion_tracking") == pytest.approx(base, rel=1e-5)
    # the world-frame term scores the same motion lower once the robot has turned
    assert score(2.0, torch.zeros(3), "motion_tracking") < 0.5 * score(0.0, torch.zeros(3), "motion_tracking")


# --- keyboard ---------------------------------------------------------------------------------


@pytest.fixture
def keyboards(monkeypatch):
    """Keyboard and PointGoalKeyboard on stand-ins for carb and Isaac Lab's Se2Keyboard."""

    class Se2Keyboard:
        def __init__(self, cfg):
            self.held = torch.zeros(3)

        def reset(self):
            self.held.zero_()

        def advance(self):
            return self.held.clone()

        def _on_keyboard_event(self, event, *args, **kwargs):
            return True

    event_type = SimpleNamespace(KEY_PRESS="press", KEY_RELEASE="release")
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(input=SimpleNamespace(KeyboardEventType=event_type)))
    monkeypatch.setitem(
        sys.modules, "isaaclab.devices", types.SimpleNamespace(Se2Keyboard=Se2Keyboard, Se2KeyboardCfg=object)
    )
    monkeypatch.delitem(sys.modules, "moveitmoveit.utils.keyboard", raising=False)
    from moveitmoveit.utils import keyboard

    yield keyboard
    sys.modules.pop("moveitmoveit.utils.keyboard", None)


def key_event(name: str, kind: str) -> SimpleNamespace:
    return SimpleNamespace(input=SimpleNamespace(name=name), type=kind)


def test_every_binary_slot_has_a_key_and_a_tap_between_advances_is_not_lost(keyboards):
    keyboard = keyboards.Keyboard(SimpleNamespace(sim_device="cpu"))
    assert set(keyboard.EXTRA_KEY_MAPPING.values()) == {
        CommandIndex.BINARY_KEY_0,
        CommandIndex.BINARY_KEY_1,
        CommandIndex.BINARY_KEY_2,
    }

    keyboard._on_keyboard_event(key_event("C", "press"))
    keyboard._on_keyboard_event(key_event("C", "release"))
    assert keyboard.advance()[CommandIndex.BINARY_KEY_2] == 1.0
    assert keyboard.advance()[CommandIndex.BINARY_KEY_2] == 0.0

    keyboard._on_keyboard_event(key_event("SPACE", "press"))
    assert [float(keyboard.advance()[CommandIndex.BINARY_KEY_1]) for _ in range(2)] == [1.0, 1.0]
    assert keyboard.advance()[CommandIndex.BEHAVIOR_PHASE] == 0.0


def test_point_goal_keyboard_also_sends_the_behavior_keys(keyboards):
    keyboard = keyboards.PointGoalKeyboard(SimpleNamespace(sim_device="cpu"), dt=0.5)
    keyboard.held[:] = torch.tensor([2.0, 0.0, 0.0])
    keyboard._on_keyboard_event(key_event("LEFT_SHIFT", "press"))

    command = keyboard.advance()
    assert command[CommandIndex.GOAL_X : CommandIndex.GOAL_Y + 1].tolist() == [2.0, 0.0]
    assert command[CommandIndex.BINARY_KEY_0] == 1.0
    assert torch.count_nonzero(command[[CommandIndex.LIN_X, CommandIndex.LIN_Y, CommandIndex.YAW]]) == 0
