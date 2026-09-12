from enum import IntEnum


class CommandIndex(IntEnum):
    LIN_X = 0
    LIN_Y = 1
    YAW = 2
    SHIFT = 3
    PUNCH = 4


COMMAND_DIM = len(CommandIndex)
