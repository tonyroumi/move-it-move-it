from dataclasses import dataclass, field
from enum import IntEnum

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


COMMAND_DIM = len(CommandIndex)


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
    """When a behavior counts as complete. Both conditions must be satisfied."""

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
    terminations: dict[str, dict[str, float]]
    """The termination terms this motion enables, each with the arguments it overrides; a term's
    other arguments keep its `@termination` defaults."""
    standing: StandingSpec | None = None
    trigger: str | None = None
    """The key slot (`binary_key_N`) whose press runs this motion once, as a behavior. None for a
    locomotion motion, which an env returns to when its behavior completes. A manifest whose motions
    all have a trigger has no locomotion to return to: its behaviors run always on, one after
    another, with no key press."""
    settle: SettleSpec | None = None
    """A behavior's stable-posture completion. None: it completes only when its clip ends."""
    loop: bool = False
    """Whether a locomotion motion's tracking target repeats its clip once it ends, instead of holding
    the last frame. A triggered behavior runs once either way: it completes at phase 1 or settles."""
    random_start: bool = True
    """Whether an env that resets into this motion's clip starts its clock at a uniform random time
    in the clip, or at time 0. Resets start only locomotion motions, or behaviors where the manifest
    declares no locomotion motion; discriminator batches still sample times uniformly."""
    amp: bool = True
    """Whether this motion earns the AMP style reward. False disables it for the motion outright,
    independent of standing mode, which disables it on its own while the robot is asked to stand."""


def load_manifest(path: str) -> list[MotionSpec]:
    """Read a manifest. A motion's position in the returned list is its motion id."""
    with open(path) as f:
        motions = yaml.safe_load(f)["motions"]
    return [_parse_motion(name, motion) for name, motion in motions.items()]


def _parse_motion(name: str, motion: dict) -> MotionSpec:
    settle = motion.get("settle")
    return MotionSpec(
        name=name,
        file=motion["file"],
        sample_prob=motion.get("sample_prob", 1.0),
        commands=_parse_commands(motion.get("commands")),
        zero_command_prob=motion.get("zero_command_prob", 0.0),
        rewards=_parse_rewards(motion.get("rewards")),
        terminations=_parse_terminations(motion.get("termination", "unhealthy")),
        standing=_parse_standing(motion["standing"]) if "standing" in motion else None,
        trigger=motion.get("trigger"),
        settle=SettleSpec(**settle) if settle else None,
        loop=motion.get("loop", False),
        random_start=motion.get("random_start", True),
        amp=motion.get("amp", True),
    )


def _parse_commands(commands: dict | None) -> dict[str, tuple[float, float]]:
    """Each command as a (low, high) range; a scalar is a fixed value."""
    return {key: tuple(value) if isinstance(value, list) else (value, value) for key, value in (commands or {}).items()}


def _parse_rewards(rewards: dict | None) -> dict[str, RewardSpec]:
    """Each reward as a weight plus kwargs; a scalar is the weight alone."""
    return {name: _parse_reward(spec) for name, spec in (rewards or {}).items()}


def _parse_reward(spec: float | dict) -> RewardSpec:
    if not isinstance(spec, dict):
        return RewardSpec(weight=spec, kwargs={})
    kwargs = dict(spec)
    return RewardSpec(weight=kwargs.pop("weight", 1.0), kwargs=kwargs)


def _parse_standing(standing: dict | None) -> StandingSpec:
    """A declared `standing` block, even an empty one, turns standing mode on with these defaults."""
    standing = standing or {}
    return StandingSpec(
        lin_vel_threshold=standing.get("lin_vel_threshold", 0.1),
        yaw_vel_threshold=standing.get("yaw_vel_threshold", 0.1),
        goal_radius=standing.get("goal_radius", 0.3),
        rewards=_parse_rewards(standing.get("rewards")),
        pose=standing.get("pose") or {},
    )


def _parse_terminations(terminations: str | list | dict) -> dict[str, dict[str, float]]:
    """A termination name or list of names, each with its default arguments, or a mapping from
    each name to the arguments it overrides (empty for the defaults)."""
    if isinstance(terminations, str):
        terminations = [terminations]
    if isinstance(terminations, list):
        return {name: {} for name in terminations}
    return {name: kwargs or {} for name, kwargs in terminations.items()}