from enum import IntEnum

import yaml

from moveitmoveit.motion.specs import MotionSpec, parse_motion


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


def load_manifest(path: str) -> list[MotionSpec]:
    """Read a manifest. A motion's position in the returned list is its motion id."""
    with open(path) as f:
        motions = yaml.safe_load(f)["motions"]
    return [parse_motion(name, motion) for name, motion in motions.items()]
