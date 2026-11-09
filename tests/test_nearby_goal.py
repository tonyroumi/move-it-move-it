"""The nearby, heading-relative goal (manifest `nearby_goal`, play's `--joystick-mode nearby_goal`),
on CPU: the env's own command and observation code runs against a fake robot (no simulator; see
conftest.py).

Run under IsaacLab's Python, which has torch: ./isaaclab.sh -p -m pytest tests
"""

import math
import sys
from itertools import product
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from conftest import quat_mul
from moveitmoveit import manifest, motion_set
from moveitmoveit.manifest import COMMAND_DIM, CommandIndex, load_manifest
from moveitmoveit.mdp import REWARDS, BodyState, ComposedMDP, StepContext
from moveitmoveit.utils import transforms


CONFIGS = Path(__file__).resolve().parents[1] / "configs"
GOAL = [CommandIndex.GOAL_X, CommandIndex.GOAL_Y]
SE2 = [CommandIndex.LIN_X, CommandIndex.LIN_Y, CommandIndex.YAW]

NEARBY = """
motions:
  walk:
    file: unused.npz
    nearby_goal:
      distance: {distance}
      idle_prob: {idle_prob}
      switch_prob: {switch_prob}
    rewards:
      progress: {{weight: 1.0, scale: 30.0}}
    standing:
      goal_radius: 0.3
      rewards:
        upright: 1.0
"""

FIXED = """
motions:
  walk:
    file: unused.npz
    commands:
      goal_x: [-5.0, 5.0]
      goal_y: [-5.0, 5.0]
    zero_command_prob: 0.25
    rewards:
      progress: {weight: 1.0, scale: 30.0}
    standing:
      goal_radius: 0.3
      rewards:
        upright: 1.0
"""


def write_manifest(tmp_path, text: str, **fields) -> str:
    path = tmp_path / "manifest.yaml"
    path.write_text(text.format(**fields) if fields else text)
    return str(path)


def nearby_manifest(tmp_path, distance=1.0, idle_prob=0.25, switch_prob=0.0) -> str:
    return write_manifest(tmp_path, NEARBY, distance=distance, idle_prob=idle_prob, switch_prob=switch_prob)


@pytest.fixture(autouse=True)
def no_clips(monkeypatch):
    """MotionSet without loading any clip files."""
    monkeypatch.setattr(motion_set, "MotionLibrary", lambda files, device: None)


def motions_from(path: str) -> motion_set.MotionSet:
    return motion_set.MotionSet(load_manifest(path), "cpu")


def yaw_quat(yaw: torch.Tensor) -> torch.Tensor:
    """xyzw quaternions for (N,) yaws."""
    zeros = torch.zeros_like(yaw)
    return torch.stack((zeros, zeros, torch.sin(yaw / 2), torch.cos(yaw / 2)), dim=-1)


def tilted(rotations: torch.Tensor, roll: float, pitch: float) -> torch.Tensor:
    """`rotations` pitched then rolled in the body frame, as a stumbling torso; its heading keeps the yaw."""
    n = rotations.shape[0]
    roll_q = torch.tensor([math.sin(roll / 2), 0.0, 0.0, math.cos(roll / 2)]).expand(n, -1)
    pitch_q = torch.tensor([0.0, math.sin(pitch / 2), 0.0, math.cos(pitch / 2)]).expand(n, -1)
    return quat_mul(quat_mul(rotations, pitch_q), roll_q)


# --- a MotionLearningEnv on a fake robot --------------------------------------------------------


@pytest.fixture
def env_module(monkeypatch):
    """`moveitmoveit.env`, imported against stand-ins for Isaac Lab's DirectRLEnv and the env cfg."""
    monkeypatch.setitem(sys.modules, "isaaclab.envs", SimpleNamespace(DirectRLEnv=object))
    env_cfg = ModuleType("moveitmoveit.env_cfg")
    env_cfg.MotionLearningEnvCfg = object
    monkeypatch.setitem(sys.modules, "moveitmoveit.env_cfg", env_cfg)
    monkeypatch.delitem(sys.modules, "moveitmoveit.env", raising=False)
    from moveitmoveit import env

    yield env
    sys.modules.pop("moveitmoveit.env", None)


class FakeEnv:
    """The parts of a MotionLearningEnv its command and observation code reads, with a robot whose
    torso pose the test sets. `env` is the real env object, built without the simulator."""

    def __init__(self, env_module, manifest_path: str, num_envs: int = 4):
        env = object.__new__(env_module.MotionLearningEnv)
        env.device = "cpu"
        env.num_envs = num_envs
        env.render_enabled = False
        env.cfg = SimpleNamespace(
            resample_commands=True, command_resample_steps=150, random_reset=True, num_amp_observations=2
        )
        env.motions = motions_from(manifest_path)
        env.mdp = ComposedMDP(env.motions.specs, ["root_height"], ["root_height"], "cpu")
        env.motion_ids = torch.zeros(num_envs, dtype=torch.long)
        env.commands = torch.zeros(num_envs, COMMAND_DIM)
        env.goal_directions = torch.zeros(num_envs, 2)
        env.standing = torch.zeros(num_envs, dtype=torch.bool)
        env.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        env.ref_body_index = 0
        env.key_body_indexes = [1, 2, 3, 4]
        env.amp_observation_size = 2
        env.amp_observation_buffer = torch.zeros(num_envs, 2, 1)
        env.extras = {}
        env.actions = torch.zeros(num_envs, 28)
        env.previous_actions = torch.zeros(num_envs, 28)

        data = SimpleNamespace(
            joint_pos=SimpleNamespace(torch=torch.zeros(num_envs, 28)),
            joint_vel=SimpleNamespace(torch=torch.zeros(num_envs, 28)),
            joint_acc=SimpleNamespace(torch=torch.zeros(num_envs, 28)),
            body_pos_w=SimpleNamespace(torch=torch.zeros(num_envs, 5, 3)),
            body_quat_w=SimpleNamespace(torch=yaw_quat(torch.zeros(num_envs)).unsqueeze(1).repeat(1, 5, 1)),
            body_lin_vel_w=SimpleNamespace(torch=torch.zeros(num_envs, 5, 3)),
            body_ang_vel_w=SimpleNamespace(torch=torch.zeros(num_envs, 5, 3)),
        )
        data.body_pos_w.torch[:, 0, 2] = 0.9
        env.robot = SimpleNamespace(data=data)
        self.env = env

    def set_pose(self, xy, yaw, rotations: torch.Tensor | None = None):
        data = self.env.robot.data
        data.body_pos_w.torch[:, 0, :2] = torch.as_tensor(xy, dtype=torch.float32)
        rotations = yaw_quat(torch.as_tensor(yaw, dtype=torch.float32).expand(self.env.num_envs)) if rotations is None else rotations
        data.body_quat_w.torch[:, 0] = rotations

    def observe(self) -> torch.Tensor:
        """The commands block of the policy observation, as `_get_observations` builds it."""
        return self.env._get_observations()[:, -COMMAND_DIM:]

    @property
    def root_xy(self) -> torch.Tensor:
        return self.env.robot.data.body_pos_w.torch[:, 0, :2]


def heading_frame(direction, yaw: float) -> torch.Tensor:
    """A heading-frame (forward, left) vector in world XY for a robot facing `yaw`."""
    forward = torch.tensor([math.cos(yaw), math.sin(yaw)])
    left = torch.tensor([-math.sin(yaw), math.cos(yaw)])
    return direction[0] * forward + direction[1] * left


# --- manifest -----------------------------------------------------------------------------------


def test_nearby_goal_block_parses_with_defaults(tmp_path):
    path = write_manifest(tmp_path, "motions:\n  walk:\n    file: unused.npz\n    nearby_goal:\n")
    (spec,) = load_manifest(path)
    assert spec.nearby_goal == manifest.NearbyGoalSpec(distance=1.0, idle_prob=0.25, switch_prob=0.0)
    assert spec.commands == {}


@pytest.mark.parametrize("distance", [0.3, 0.2, 0.0])
def test_a_distance_within_the_standing_goal_radius_is_an_error(tmp_path, distance):
    with pytest.raises(ValueError, match="must exceed the standing goal_radius"):
        load_manifest(nearby_manifest(tmp_path, distance=distance))


def test_nearby_goal_and_a_world_goal_command_are_exclusive(tmp_path):
    text = NEARBY.replace("    nearby_goal:", "    commands: {{goal_x: [-1.0, 1.0]}}\n    nearby_goal:")
    with pytest.raises(ValueError, match="not both"):
        load_manifest(write_manifest(tmp_path, text, distance=1.0, idle_prob=0.25, switch_prob=0.0))


@pytest.mark.parametrize("key", ["idle_prob", "switch_prob"])
def test_nearby_goal_probabilities_must_be_probabilities(tmp_path, key):
    with pytest.raises(ValueError, match=f"{key} must be in"):
        load_manifest(nearby_manifest(tmp_path, **{key: 1.5}))


def test_unknown_nearby_goal_keys_are_an_error(tmp_path):
    path = write_manifest(tmp_path, "motions:\n  walk:\n    file: unused.npz\n    nearby_goal: {speed: 1.0}\n")
    with pytest.raises(ValueError, match="unknown"):
        load_manifest(path)


def test_shipped_manifests():
    (nearby,) = load_manifest(str(CONFIGS / "nearby_goal_manifest.yaml"))
    assert nearby.nearby_goal.distance > nearby.standing.goal_radius
    assert 0.0 < nearby.nearby_goal.switch_prob < 1.0
    ComposedMDP([nearby], ["root_height"], ["root_height"], "cpu", joint_names=JOINT_NAMES)

    # the fixed world-goal manifest is unchanged
    (fixed,) = load_manifest(str(CONFIGS / "goal_manifest.yaml"))
    assert fixed.nearby_goal is None and {"goal_x", "goal_y"} <= set(fixed.commands)


JOINT_NAMES = [
    f"{side}_{joint}"
    for side in ("right", "left")
    for joint in ("hip_x", "ankle_x", "shoulder_x", "elbow")
] + [f"joint_{i}" for i in range(20)]


# --- heading transforms -------------------------------------------------------------------------


def test_heading_to_world_inverts_world_to_heading_even_for_a_tilted_body():
    yaw = torch.linspace(-math.pi, math.pi, 9)
    rotations = tilted(yaw_quat(yaw), roll=0.3, pitch=-0.4)
    vectors = torch.randn(9, 2)

    world = transforms.heading_to_world_xy(rotations, vectors)
    assert torch.allclose(transforms.world_to_heading_frame_xy(rotations, world), vectors, atol=1e-5)
    # yaw only: the tilt doesn't shorten or tip the vector
    assert torch.allclose(world.norm(dim=-1), vectors.norm(dim=-1), atol=1e-5)


# --- keyboard -----------------------------------------------------------------------------------


ARROWS = {"up": (1.5, 0.0), "down": (-1.5, 0.0), "left": (0.0, 1.0), "right": (0.0, -1.0)}


def held(*keys) -> torch.Tensor:
    """The SE(2) command Isaac Lab's Se2Keyboard holds for these arrow keys (sensitivities 1.5/1.0)."""
    command = torch.zeros(3)
    for key in keys:
        command[:2] += torch.tensor(ARROWS[key])
    return command


@pytest.mark.parametrize(
    "keys, direction",
    [
        ((), (0.0, 0.0)),
        (("up",), (1.0, 0.0)),
        (("down",), (-1.0, 0.0)),
        (("left",), (0.0, 1.0)),
        (("right",), (0.0, -1.0)),
        (("up", "left"), (math.sqrt(0.5), math.sqrt(0.5))),
        (("down", "right"), (-math.sqrt(0.5), -math.sqrt(0.5))),
        (("up", "down"), (0.0, 0.0)),
        (("up", "down", "left"), (0.0, 1.0)),
    ],
)
def test_nearby_goal_keyboard_holds_a_normalized_heading_frame_direction(keyboards, keys, direction):
    keyboard = keyboards.NearbyGoalKeyboard(SimpleNamespace(sim_device="cpu"))
    keyboard.held[:] = held(*keys)
    keyboard.held[2] = 0.7  # a yaw key held too: ignored

    command = keyboard.advance()
    assert torch.allclose(command[GOAL], torch.tensor(direction), atol=1e-6)
    assert torch.count_nonzero(command[SE2]) == 0  # no velocity command reaches the policy


def test_nearby_goal_keyboard_releases_to_no_direction_and_keeps_the_binary_keys(keyboards):
    keyboard = keyboards.NearbyGoalKeyboard(SimpleNamespace(sim_device="cpu"))
    keyboard.held[:] = held("up")
    keyboard._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name="LEFT_SHIFT"), type="press"))
    command = keyboard.advance()
    assert command[GOAL].tolist() == [1.0, 0.0] and command[CommandIndex.BINARY_KEY_0] == 1.0

    keyboard.held.zero_()  # up released
    keyboard._on_keyboard_event(SimpleNamespace(input=SimpleNamespace(name="LEFT_SHIFT"), type="release"))
    command = keyboard.advance()
    assert command[GOAL].tolist() == [0.0, 0.0] and command[CommandIndex.BINARY_KEY_0] == 0.0


def test_keyboard_directions_are_the_ones_training_samples(keyboards):
    """Training/play parity: every non-zero arrow-key combination is one of the trained directions."""
    keyboard = keyboards.NearbyGoalKeyboard(SimpleNamespace(sim_device="cpu"))
    trained = motion_set.NEARBY_GOAL_DIRECTIONS
    seen = set()
    for pressed in product([False, True], repeat=4):
        keyboard.held[:] = held(*(key for key, on in zip(ARROWS, pressed) if on))
        direction = keyboard.advance()[GOAL]
        if direction.norm() == 0:
            continue
        match = torch.isclose(trained, direction, atol=1e-6).all(dim=-1).nonzero()
        assert match.numel() == 1, direction
        seen.add(int(match))
    assert seen == set(range(len(trained)))


# --- the env: placing and observing the goal ----------------------------------------------------


DIRECTIONS = {
    "forward": (1.0, 0.0),
    "back": (-1.0, 0.0),
    "left": (0.0, 1.0),
    "right": (0.0, -1.0),
    "forward-left": (math.sqrt(0.5), math.sqrt(0.5)),
}


@pytest.mark.parametrize("name", DIRECTIONS)
@pytest.mark.parametrize("yaw", [0.0, 0.6, math.pi / 2, -2.5, math.pi])
def test_held_direction_puts_the_goal_that_far_along_the_robots_heading(env_module, tmp_path, name, yaw):
    fake = FakeEnv(env_module, nearby_manifest(tmp_path, distance=1.2))
    fake.env.cfg.resample_commands = False  # play: the keyboard holds the direction
    fake.set_pose((3.0, -1.0), yaw)
    direction = torch.tensor(DIRECTIONS[name])
    fake.env.goal_directions[:] = direction

    observed = fake.observe()

    expected = torch.tensor([3.0, -1.0]) + 1.2 * heading_frame(direction, yaw)
    assert torch.allclose(fake.env.commands[:, GOAL], expected.expand(4, -1), atol=1e-5)
    assert torch.allclose((fake.env.commands[:, GOAL] - fake.root_xy).norm(dim=-1), torch.full((4,), 1.2), atol=1e-5)
    # no double transform: the policy observes exactly the held heading-frame offset
    assert torch.allclose(observed[:, GOAL], (1.2 * direction).expand(4, -1), atol=1e-5)
    assert not fake.env.standing.any()


def test_the_goal_follows_the_robot_as_it_moves_and_turns(env_module, tmp_path):
    fake = FakeEnv(env_module, nearby_manifest(tmp_path))
    fake.env.cfg.resample_commands = False
    fake.env.goal_directions[:] = torch.tensor([0.0, 1.0])  # left held

    path = [((0.0, 0.0), 0.0), ((0.3, 0.1), 0.2), ((0.5, 0.4), 1.0), ((0.4, 0.9), 2.4)]
    for i, (xy, yaw) in enumerate(path):
        rotations = tilted(yaw_quat(torch.full((4,), yaw)), roll=0.1 * i, pitch=0.05)
        fake.set_pose(xy, yaw, rotations)
        observed = fake.observe()

        expected = torch.tensor(xy) + heading_frame((0.0, 1.0), yaw)
        assert torch.allclose(fake.env.commands[:, GOAL], expected.expand(4, -1), atol=1e-5)
        assert torch.allclose(observed[:, GOAL], torch.tensor([0.0, 1.0]).expand(4, -1), atol=1e-5)


def test_no_direction_puts_the_goal_at_the_robot_and_asks_it_to_stand(env_module, tmp_path):
    fake = FakeEnv(env_module, nearby_manifest(tmp_path))
    fake.env.cfg.resample_commands = False
    fake.set_pose((2.0, 1.0), 0.8)

    fake.env.goal_directions[:] = torch.tensor([1.0, 0.0])
    fake.observe()
    assert not fake.env.standing.any()

    fake.env.goal_directions.zero_()  # key released: takes effect on the very next placement
    fake.env.place_nearby_goals()
    assert torch.allclose(fake.env.commands[:, GOAL], fake.root_xy)

    observed = fake.observe()
    assert torch.count_nonzero(observed[:, GOAL]) == 0
    assert fake.env.standing.all()


def test_play_overrides_the_distance_but_not_below_the_goal_radius(env_module, tmp_path):
    fake = FakeEnv(env_module, nearby_manifest(tmp_path, distance=1.0))
    with pytest.raises(ValueError, match="must exceed the standing goal_radius"):
        fake.env.motions.use_nearby_goals(0.3)

    fake.env.motions.use_nearby_goals(0.5)
    fake.env.cfg.resample_commands = False
    fake.env.goal_directions[:] = torch.tensor([1.0, 0.0])
    observed = fake.observe()
    assert torch.allclose(observed[:, GOAL], torch.tensor([0.5, 0.0]).expand(4, -1), atol=1e-6)
    assert not fake.env.standing.any()


# --- the env: training's sampler ----------------------------------------------------------------


def test_training_samples_none_and_every_direction(tmp_path):
    motions = motions_from(nearby_manifest(tmp_path, idle_prob=0.25))
    torch.manual_seed(0)
    directions = motions.sample_goal_directions(torch.zeros(20000, dtype=torch.long))

    norms = directions.norm(dim=-1)
    idle = norms == 0
    assert torch.allclose(norms[~idle], torch.ones(1), atol=1e-6)  # unit, diagonals included
    assert abs(idle.float().mean().item() - 0.25) < 0.02
    angles = torch.atan2(directions[~idle, 1], directions[~idle, 0])
    counts = torch.bincount(torch.round(angles / (math.pi / 4)).long() % 8, minlength=8)
    assert (counts > 0.9 * counts.float().mean()).all()


def test_training_recomputes_the_goal_every_step_and_switches_directions(env_module, tmp_path):
    fake = FakeEnv(env_module, nearby_manifest(tmp_path, idle_prob=0.0, switch_prob=1.0), num_envs=64)
    torch.manual_seed(0)

    fake.observe()  # episode start: directions sampled and goals placed
    first = fake.env.goal_directions.clone()
    assert torch.allclose(first.norm(dim=-1), torch.ones(64), atol=1e-6)

    for step in range(1, 6):
        fake.env.episode_length_buf += 1
        fake.set_pose((0.1 * step, 0.05 * step), 0.3 * step)
        context = fake.env._step_context()
        fake.env.mdp.rewards(context, fake.env.motion_ids, fake.env.standing)
        observed = fake.observe()

        # re-placed from the new pose, and observed as distance * the (possibly new) direction
        offsets = fake.env.goal_directions * 1.0
        assert torch.allclose(observed[:, GOAL], offsets, atol=1e-5)
        expected = fake.root_xy + transforms.heading_to_world_xy(fake.env.robot.data.body_quat_w.torch[:, 0], offsets)
        assert torch.allclose(fake.env.commands[:, GOAL], expected, atol=1e-5)
        # a direction switch isn't a resample: progress keeps its memory
        assert not torch.isnan(fake.env.mdp._memory["root_xy"]).any()

    assert (fake.env.goal_directions != first).any(dim=-1).any()  # transitions happened


def test_progress_toward_a_nearby_goal_is_motion_along_the_held_direction(env_module, tmp_path):
    fake = FakeEnv(env_module, nearby_manifest(tmp_path), num_envs=1)
    fake.env.cfg.resample_commands = False
    yaw = 0.7
    fake.set_pose((0.0, 0.0), yaw)
    fake.env.goal_directions[:] = torch.tensor([1.0, 0.0])
    fake.observe()

    def step_to(xy):
        fake.set_pose(xy, yaw)
        _, raw, _ = fake.env.mdp.rewards(fake.env._step_context(), fake.env.motion_ids, fake.env.standing)
        fake.observe()
        return raw["progress"].item()

    step_to((0.0, 0.0))  # first step: no memory yet, 0
    forward = heading_frame((0.05, 0.0), yaw)
    assert step_to(tuple(forward.tolist())) == pytest.approx(30.0 * 0.05, rel=1e-3)
    back = forward - heading_frame((0.05, 0.0), yaw)
    assert step_to(tuple(back.tolist())) == pytest.approx(-30.0 * 0.05, rel=1e-3)


# --- existing modes -----------------------------------------------------------------------------


def test_fixed_world_goals_stay_put_and_get_no_direction(env_module, tmp_path):
    fake = FakeEnv(env_module, write_manifest(tmp_path, FIXED))
    torch.manual_seed(0)
    fake.observe()
    goals = fake.env.commands[:, GOAL].clone()
    assert torch.count_nonzero(fake.env.goal_directions) == 0
    assert not fake.env.motions.nearby_goal.any()

    fake.env.episode_length_buf += 1
    fake.set_pose((1.0, 2.0), 1.3)
    fake.observe()
    assert torch.equal(fake.env.commands[:, GOAL], goals)  # fixed in world space while the robot moves


def test_progress_toward_a_fixed_goal_is_the_change_in_distance(tmp_path):
    (spec,) = load_manifest(write_manifest(tmp_path, FIXED))
    mdp = ComposedMDP([spec], ["root_height"], ["root_height"], "cpu")
    commands = torch.zeros(2, COMMAND_DIM)
    commands[:, GOAL] = torch.tensor([[3.0, 4.0], [-1.0, 0.0]])

    def progress(root_xy) -> torch.Tensor:
        root = torch.zeros(2, 3)
        root[:, :2] = torch.tensor(root_xy)
        state = BodyState(*(torch.zeros(2, 1),) * 2, root, yaw_quat(torch.zeros(2)), *(torch.zeros(2, 3),) * 2, None)
        c = StepContext(state, None, commands, None, None, lambda: None)
        c.memory = mdp._memory
        return REWARDS["progress"].fn(c, scale=torch.ones(2))

    assert torch.count_nonzero(progress([[0.0, 0.0], [0.0, 0.0]])) == 0
    # distances 5 -> 4 and 1 -> 2
    assert torch.allclose(progress([[0.6, 0.8], [1.0, 0.0]]), torch.tensor([1.0, -1.0]), atol=1e-5)
    mdp.reset_memory(torch.tensor([0]))  # a new goal for env 0 earns nothing on its first step
    assert torch.allclose(progress([[1.2, 1.6], [0.5, 0.0]]), torch.tensor([0.0, 0.5]), atol=1e-5)


def test_point_goal_mode_keeps_the_keyboards_world_goal_for_a_nearby_checkpoint(env_module, tmp_path):
    fake = FakeEnv(env_module, nearby_manifest(tmp_path))
    fake.env.motions.use_world_goals()
    fake.env.cfg.resample_commands = False
    fake.env.commands[:, GOAL] = torch.tensor([4.0, 4.0])
    fake.env.goal_directions[:] = torch.tensor([1.0, 0.0])

    fake.observe()
    assert torch.equal(fake.env.commands[:, GOAL], torch.tensor([4.0, 4.0]).expand(4, -1))


def test_nearby_mode_works_for_a_fixed_goal_checkpoint_at_the_default_distance(env_module, tmp_path):
    fake = FakeEnv(env_module, write_manifest(tmp_path, FIXED))
    fake.env.motions.use_nearby_goals()
    fake.env.cfg.resample_commands = False
    fake.env.goal_directions[:] = torch.tensor([0.0, -1.0])

    observed = fake.observe()
    distance = manifest.DEFAULT_NEARBY_GOAL_DISTANCE
    assert torch.allclose(observed[:, GOAL], torch.tensor([0.0, -distance]).expand(4, -1), atol=1e-6)
