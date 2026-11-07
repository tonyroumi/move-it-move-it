"""Parses and validates a motion manifest. Deliberately has no torch dependency."""

from dataclasses import dataclass
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
MOTION_KEYS = frozenset({"file", "sample_prob", "commands", "zero_command_prob", "rewards", "termination"})


@dataclass(frozen=True)
class RewardSpec:
    weight: float
    kwargs: dict[str, float]


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


def load_manifest(path: str) -> list[MotionSpec]:
    """Read a manifest. A motion's position in the returned list is its motion id.

    Raises:
        ValueError: If the manifest is malformed or names anything unknown.
    """
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

    return MotionSpec(
        name=name,
        file=motion["file"],
        sample_prob=sample_prob,
        commands=_parse_commands(motion.get("commands"), name),
        zero_command_prob=zero_command_prob,
        rewards=_parse_rewards(motion.get("rewards"), name),
        terminations=_parse_terminations(motion.get("termination", "unhealthy"), name),
    )


def _parse_commands(commands: Any, motion: str) -> dict[str, tuple[float, float]]:
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


def _parse_terminations(terminations: Any, motion: str) -> tuple[str, ...]:
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
    if not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {value!r}.")
    return float(value)


def check_known(names: Any, known: Any, where: str) -> None:
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"{where}: unknown {unknown}. Known: {sorted(known)}.")
