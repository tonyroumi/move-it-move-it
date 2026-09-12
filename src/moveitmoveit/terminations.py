from enum import IntEnum


class TerminationIndex(IntEnum):
    DEVIATION_FROM_MOTION = 0
    UNHEALTHY = 1


NUM_TERMINATIONS = len(TerminationIndex)
