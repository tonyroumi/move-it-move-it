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
    BINARY_KEY_2 = 7
    BEHAVIOR_PHASE = 8
    """Progress through the running behavior, which the env fills in. No motion declares it."""


COMMAND_DIM = len(CommandIndex)
COMMAND_KEYS = frozenset(command.name.lower() for command in CommandIndex if command != CommandIndex.BEHAVIOR_PHASE)
TRIGGER_KEYS = frozenset(key for key in COMMAND_KEYS if key.startswith("binary_key_"))
MOTION_KEYS = frozenset(
    {"file", "sample_prob", "commands", "zero_command_prob", "rewards", "termination", "standing", "style_condition", "trigger", "settle"}
)

STYLE_CONDITION_COMMAND = CommandIndex.BINARY_KEY_0
"""The command a conditional discriminator's behavior condition comes from: the binary key."""
NUM_STYLE_CONDITIONS = 2
"""Key off (0, e.g. walk) and key on (1, e.g. run)."""
STANDING_KEYS = frozenset({"lin_vel_threshold", "yaw_vel_threshold", "goal_radius", "pose", "rewards"})
SETTLE_KEYS = frozenset({"min_phase", "max_dof_speed"})


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
class SettleSpec:
    """When a behavior counts as complete before its clip ends: once it is `min_phase` of the way
    through and the robot's RMS joint speed is below `max_dof_speed` (rad/s)."""

    min_phase: float
    max_dof_speed: float


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
    style_condition: int | None = None
    """The behavior condition this clip's frames are real examples of, for a conditional
    discriminator (0 or 1, the value of its binary key). None in an unconditional manifest."""
    trigger: str | None = None
    """The key slot (`binary_key_N`) whose press runs this motion once, as a behavior. None for a
    locomotion motion, which an env returns to when its behavior completes."""
    settle: SettleSpec | None = None
    """A behavior's stable-posture completion. None: it completes only when its clip ends."""


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
    check_style_conditions(specs, path)
    _check_triggers(specs, path)
    return specs


def generated_style_conditions(spec: MotionSpec) -> set[int]:
    """The conditions an env on this motion can label its own (generated) AMP samples with. Its
    binary key is fixed at its `style_condition`; a zeroed command turns the key off, but with a
    `standing` block a zeroed command always means standing, whose samples carry no condition."""
    conditions = {spec.style_condition}
    if spec.zero_command_prob > 0.0 and spec.standing is None:
        conditions.add(0)
    return conditions


def check_style_conditions(specs: list[MotionSpec], where: str) -> None:
    """A manifest is conditional when its motions declare `style_condition`. Then every motion must,
    each one's binary key must be fixed at its condition, and every condition an env can generate
    must have a real clip to compare against."""
    declared = [spec.style_condition is not None for spec in specs]
    if not any(declared):
        return
    if not all(declared):
        missing = [spec.name for spec in specs if spec.style_condition is None]
        raise ValueError(f"{where}: style_condition must be declared by every motion or by none; missing in {missing}.")

    key = STYLE_CONDITION_COMMAND.name.lower()
    for spec in specs:
        low, high = spec.commands.get(key, (0.0, 0.0))
        if not low == high == float(spec.style_condition):
            raise ValueError(
                f"{spec.name}: style_condition {spec.style_condition} needs commands.{key} fixed at"
                f" {spec.style_condition}, got [{low}, {high}]. Generated samples are labeled from that key,"
                " so the two must agree."
            )

    real = {spec.style_condition for spec in specs if spec.sample_prob > 0.0}
    for spec in specs:
        if spec.sample_prob > 0.0:
            unmatched = sorted(generated_style_conditions(spec) - real)
            if unmatched:
                raise ValueError(
                    f"{spec.name}: its envs generate style_condition {unmatched}, but no motion with a positive"
                    " sample_prob provides real clips for it. A zeroed command turns the key off: declare a"
                    " standing block, set zero_command_prob to 0, or add such a clip."
                )


def _check_triggers(specs: list[MotionSpec], path: str) -> None:
    behaviors = [spec for spec in specs if spec.trigger is not None]
    if not behaviors:
        return
    if any(spec.style_condition is not None for spec in specs):
        raise ValueError(
            f"{path}: triggered behaviors can't be combined with style_condition. A running behavior is"
            " observed in the binary key slots, which a conditional discriminator reads its labels from."
        )

    triggers = [spec.trigger for spec in behaviors]
    duplicates = sorted({key for key in triggers if triggers.count(key) > 1})
    if duplicates:
        raise ValueError(f"{path}: {duplicates} each trigger more than one motion.")

    for spec in specs:
        driven = sorted(set(spec.commands) & set(triggers))
        if driven:
            raise ValueError(f"{spec.name}.commands: {driven} trigger behaviors, so no motion may declare them.")

    if not any(spec.trigger is None and spec.sample_prob > 0 for spec in specs):
        raise ValueError(f"{path}: behaviors need a locomotion motion (no trigger) with a positive sample_prob.")


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

    trigger = motion.get("trigger")
    if trigger is not None:
        if not isinstance(trigger, str) or trigger not in TRIGGER_KEYS:
            raise ValueError(f"{name}.trigger: expected one of {sorted(TRIGGER_KEYS)}, got {trigger!r}.")
        if commands:
            raise ValueError(f"{name}: a triggered behavior keeps its locomotion motion's commands; declare none.")
        if not sample_prob > 0:
            raise ValueError(f"{name}: a triggered behavior needs a positive sample_prob, which sets how often training triggers it.")
        if "standing" in motion:
            raise ValueError(f"{name}.standing: a triggered behavior runs to completion, so it has no standing mode.")

    settle = motion.get("settle")
    if settle is not None:
        if trigger is None:
            raise ValueError(f"{name}.settle: only a triggered behavior can settle.")
        settle = _parse_settle(settle, name)

    return MotionSpec(
        name=name,
        file=motion["file"],
        sample_prob=sample_prob,
        commands=commands,
        zero_command_prob=zero_command_prob,
        rewards=_parse_rewards(motion.get("rewards"), name),
        terminations=_parse_terminations(motion.get("termination", "unhealthy"), name),
        standing=_parse_standing(motion["standing"], name) if "standing" in motion else None,
        style_condition=_parse_style_condition(motion["style_condition"], name) if "style_condition" in motion else None,
        trigger=trigger,
        settle=settle,
    )


def _parse_style_condition(value: Any, motion: str) -> int:
    """A clip's behavior condition: 0 (key off) or 1 (key on)."""
    if isinstance(value, bool) or not isinstance(value, int) or value not in range(NUM_STYLE_CONDITIONS):
        raise ValueError(f"{motion}.style_condition: expected one of {list(range(NUM_STYLE_CONDITIONS))}, got {value!r}.")
    return value


def _parse_settle(settle: Any, motion: str) -> SettleSpec:
    settle = _mapping(settle, f"{motion}.settle")
    check_known(settle, SETTLE_KEYS, f"{motion}.settle")
    missing = sorted(SETTLE_KEYS - set(settle))
    if missing:
        raise ValueError(f"{motion}.settle: missing {missing}.")

    min_phase = _number(settle["min_phase"], f"{motion}.settle.min_phase")
    max_dof_speed = _number(settle["max_dof_speed"], f"{motion}.settle.max_dof_speed")
    if not 0.0 < min_phase <= 1.0:
        raise ValueError(f"{motion}.settle.min_phase: must be in (0, 1], got {min_phase}.")
    if not max_dof_speed > 0.0:
        raise ValueError(f"{motion}.settle.max_dof_speed: must be positive, got {max_dof_speed}.")
    return SettleSpec(min_phase, max_dof_speed)


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
