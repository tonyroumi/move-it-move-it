from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import yaml


class CommandIndex(IntEnum):
    LIN_X = 0
    LIN_Y = 1
    YAW = 2
    BINARY_KEY_0 = 3
    BINARY_KEY_1 = 4
    GOAL_X = 5
    GOAL_Y = 6


COMMAND_DIM = len(CommandIndex)
COMMAND_KEYS = frozenset(command.name.lower() for command in CommandIndex)
MOTION_KEYS = frozenset(
    {"file", "sample_prob", "commands", "zero_command_prob", "rewards", "termination", "standing", "nearby_goal"}
)
STANDING_KEYS = frozenset({"lin_vel_threshold", "yaw_vel_threshold", "goal_radius", "pose", "rewards"})
NEARBY_GOAL_KEYS = frozenset({"distance", "idle_prob", "switch_prob"})

# m; play's nearby_goal joystick mode falls back to it for a checkpoint whose manifest declares none
DEFAULT_NEARBY_GOAL_DISTANCE = 1.0


@dataclass(frozen=True)
class RewardSpec:
    weight: float
    kwargs: dict[str, float]


@dataclass(frozen=True)
class StandingSpec:
    """A motion's standing mode: the commands that ask the robot to stand still, and the rewards
    that replace the motion's own `rewards` (with the AMP style reward off) while they do."""

    lin_vel_threshold: float
    yaw_vel_threshold: float
    goal_radius: float
    rewards: dict[str, RewardSpec]
    pose: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class NearbyGoalSpec:
    """A motion's nearby, heading-relative goal: instead of a fixed world point, the goal is placed
    `distance` m from the root, in the root's heading frame, along a direction held like an arrow key
    (forward/back/left/right or a normalized diagonal), and recomputed from the root's current
    position and heading every step. No direction puts the goal at the root, i.e. stand still."""

    distance: float
    idle_prob: float
    """Probability that a sampled direction is none."""
    switch_prob: float
    """Per-step probability of a new direction between the regular command resamples."""


@dataclass(frozen=True)
class MotionSpec:
    name: str
    file: str
    sample_prob: float
    commands: dict[str, tuple[float, float]]
    """(low, high) per declared command. Absent commands are always 0."""
    zero_command_prob: float
    rewards: dict[str, RewardSpec]
    terminations: tuple[str, ...]
    standing: StandingSpec | None = None
    nearby_goal: NearbyGoalSpec | None = None


def load_manifest(path: str) -> list[MotionSpec]:
    """Read a manifest. A motion's position in the returned list is its motion id."""
    with open(path, "r") as f:
        document = yaml.safe_load(f)

    if not isinstance(document, dict) or set(document) != {"motions"}:
        raise ValueError(f"{path}: expected a single top-level 'motions' key.")
    motions = _mapping(document["motions"], "motions")
    if not motions:
        raise ValueError(f"{path}: declares no motions.")

    specs = [_parse_motion(name, motion) for name, motion in motions.items()]
    if not sum(spec.sample_prob for spec in specs) > 0:
        raise ValueError(f"{path}: at least one motion needs a positive sample_prob.")
    return specs


def _parse_motion(name: str, motion: Any) -> MotionSpec:
    """Validate one motion entry."""
    motion = _mapping(motion, name)
    check_known(motion, MOTION_KEYS, name)
    if not isinstance(motion.get("file"), str):
        raise ValueError(f"{name}: 'file' must be a path.")

    sample_prob = _number(motion.get("sample_prob", 1.0), f"{name}.sample_prob")
    if sample_prob < 0.0:
        raise ValueError(f"{name}: sample_prob must not be negative, got {sample_prob}.")

    zero_command_prob = _number(motion.get("zero_command_prob", 0.0), f"{name}.zero_command_prob")
    if not 0.0 <= zero_command_prob <= 1.0:
        raise ValueError(f"{name}: zero_command_prob must be in [0, 1], got {zero_command_prob}.")

    commands = _parse_commands(motion.get("commands"), name)
    standing = _parse_standing(motion["standing"], name) if "standing" in motion else None
    nearby_goal = None
    if "nearby_goal" in motion:
        if any(key in commands for key in ("goal_x", "goal_y")):
            raise ValueError(
                f"{name}: declare either commands.goal_x/goal_y (a fixed world-space goal) or nearby_goal"
                " (a goal placed relative to the root every step), not both."
            )
        nearby_goal = _parse_nearby_goal(motion["nearby_goal"], name, standing)

    return MotionSpec(
        name=name,
        file=motion["file"],
        sample_prob=sample_prob,
        commands=commands,
        zero_command_prob=zero_command_prob,
        rewards=_parse_rewards(motion.get("rewards"), name),
        terminations=_parse_terminations(motion.get("termination", "unhealthy"), name),
        standing=standing,
        nearby_goal=nearby_goal,
    )


def _parse_commands(commands: Any, motion: str) -> dict[str, tuple[float, float]]:
    """Each command as a (low, high) range; a scalar is a fixed value."""
    commands = _mapping(commands, f"{motion}.commands")
    check_known(commands, COMMAND_KEYS, f"{motion}.commands")

    parsed = {}
    for key, value in commands.items():
        where = f"{motion}.commands.{key}"
        if isinstance(value, list):
            if len(value) != 2:
                raise ValueError(f"{where}: a range must be [low, high], got {value}.")
            low, high = _number(value[0], where), _number(value[1], where)
            if low > high:
                raise ValueError(f"{where}: reversed range {value}.")
        else:
            low = high = _number(value, where)
        parsed[key] = (low, high)
    return parsed


def _parse_rewards(rewards: Any, motion: str) -> dict[str, RewardSpec]:
    """Each reward as a weight plus kwargs; a scalar is the weight alone."""
    rewards = _mapping(rewards, f"{motion}.rewards")

    parsed = {}
    for name, spec in rewards.items():
        where = f"{motion}.rewards.{name}"
        if isinstance(spec, dict):
            kwargs = {key: value for key, value in spec.items() if key != "weight"}
            parsed[name] = RewardSpec(
                weight=_number(spec.get("weight", 1.0), f"{where}.weight"),
                kwargs={key: _number(value, f"{where}.{key}") for key, value in kwargs.items()},
            )
        else:
            parsed[name] = RewardSpec(weight=_number(spec, where), kwargs={})
    return parsed


def _parse_standing(standing: Any, motion: str) -> StandingSpec:
    """A declared `standing` block, even an empty one, turns standing mode on with these defaults."""
    standing = _mapping(standing, f"{motion}.standing")
    check_known(standing, STANDING_KEYS, f"{motion}.standing")

    thresholds = {}
    for key, default in (("lin_vel_threshold", 0.1), ("yaw_vel_threshold", 0.1), ("goal_radius", 0.3)):
        thresholds[key] = _number(standing.get(key, default), f"{motion}.standing.{key}")
        if thresholds[key] < 0.0:
            raise ValueError(f"{motion}.standing: {key} must not be negative, got {thresholds[key]}.")

    pose = _mapping(standing.get("pose"), f"{motion}.standing.pose")
    pose = {joint: _number(value, f"{motion}.standing.pose.{joint}") for joint, value in pose.items()}

    return StandingSpec(**thresholds, rewards=_parse_rewards(standing.get("rewards"), f"{motion}.standing"), pose=pose)


def _parse_nearby_goal(nearby_goal: Any, motion: str, standing: StandingSpec | None) -> NearbyGoalSpec:
    """A declared `nearby_goal` block, even an empty one, turns the nearby goal on with these defaults."""
    nearby_goal = _mapping(nearby_goal, f"{motion}.nearby_goal")
    check_known(nearby_goal, NEARBY_GOAL_KEYS, f"{motion}.nearby_goal")

    distance = _number(nearby_goal.get("distance", DEFAULT_NEARBY_GOAL_DISTANCE), f"{motion}.nearby_goal.distance")
    check_nearby_goal_distance(distance, standing.goal_radius if standing is not None else 0.0, f"{motion}.nearby_goal")

    probs = {}
    for key, default in (("idle_prob", 0.25), ("switch_prob", 0.0)):
        probs[key] = _number(nearby_goal.get(key, default), f"{motion}.nearby_goal.{key}")
        if not 0.0 <= probs[key] <= 1.0:
            raise ValueError(f"{motion}.nearby_goal: {key} must be in [0, 1], got {probs[key]}.")

    return NearbyGoalSpec(distance=distance, **probs)


def check_nearby_goal_distance(distance: float, goal_radius: float, where: str) -> None:
    """A held direction must put the goal outside the standing radius, or it would still mean "stand"."""
    if not distance > goal_radius:
        raise ValueError(
            f"{where}: the nearby goal distance ({distance} m) must exceed the standing goal_radius"
            f" ({goal_radius} m), or a held direction would still read as standing still."
        )


def _parse_terminations(terminations: Any, motion: str) -> tuple[str, ...]:
    """A termination name or list of names."""
    if isinstance(terminations, str):
        terminations = [terminations]
    if not isinstance(terminations, list):
        raise ValueError(f"{motion}.termination: expected a name or a list of names, got {terminations!r}.")
    if not all(isinstance(name, str) for name in terminations):
        raise ValueError(f"{motion}.termination: expected a name or a list of names, got {terminations!r}.")
    return tuple(terminations)


def _mapping(value: Any, where: str) -> dict:
    """A key left empty in YAML (`rewards:`) reads as None, which means empty."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected a mapping, got {value!r}.")
    return value


def _number(value: Any, where: str) -> float:
    """`value` as a float, or a ValueError naming `where`."""
    if not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}.")
    return float(value)


def check_known(names: Any, known: Any, where: str) -> None:
    """Raise a ValueError naming `where` if any of `names` is not in `known`."""
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"{where}: unknown {unknown}. Known: {sorted(known)}.")
