"""The specs a manifest motion entry declares."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RewardSpec:
    weight: float
    kwargs: dict[str, float]


@dataclass(frozen=True)
class MotionSpec:
    name: str
    file: str
    duration: float
    sample_prob: float
    commands: dict[str, tuple[float, float]]
    zero_command_prob: float
    rewards: dict[str, RewardSpec]
    terminations: dict[str, dict[str, float]]
    trigger: str | None = None
    loop: bool = False
    random_start: bool = True
    amp: float = 1.0


def parse_motion(name: str, motion: dict, duration: float) -> MotionSpec:
    return MotionSpec(
        name=name,
        file=motion["file"],
        sample_prob=motion.get("sample_prob", 1.0),
        commands=_parse_commands(motion.get("commands")),
        zero_command_prob=motion.get("zero_command_prob", 0.0),
        rewards=_parse_rewards(motion.get("rewards")),
        terminations=_parse_terminations(motion.get("termination", "unhealthy")),
        trigger=motion.get("trigger"),
        loop=motion.get("loop", False),
        random_start=motion.get("random_start", True),
        amp=motion.get("amp", 1.0),
        duration=duration,
    )


def _parse_commands(commands: dict | None) -> dict[str, tuple[float, float]]:
    return {key: tuple(value) if isinstance(value, list) else (value, value) for key, value in (commands or {}).items()}


def _parse_rewards(rewards: dict | None) -> dict[str, RewardSpec]:
    parsed = {}
    for name, spec in (rewards or {}).items():
        kwargs = dict(spec) if isinstance(spec, dict) else {"weight": spec}
        parsed[name] = RewardSpec(weight=kwargs.pop("weight", 1.0), kwargs=kwargs)
    return parsed


def _parse_terminations(terminations: str | list | dict) -> dict[str, dict[str, float]]:
    return {name: kwargs or {} for name, kwargs in terminations.items()}
