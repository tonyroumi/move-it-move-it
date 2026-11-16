import yaml

from moveitmoveit.mdp.commands import COMMAND_DIM, CommandIndex
from moveitmoveit.motion.specs import MotionSpec, parse_motion

__all__ = ["COMMAND_DIM", "CommandIndex", "load_manifest"]


def load_manifest(path: str, durations: list[float]) -> list[MotionSpec]:
    """Read a manifest. A motion's position in the returned list is its motion id."""
    with open(path) as f:
        motions = yaml.safe_load(f)["motions"]
    return [parse_motion(name, motion, durations[i]) for i, (name, motion) in enumerate(motions.items())]
