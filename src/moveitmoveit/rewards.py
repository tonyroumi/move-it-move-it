from enum import IntEnum


class RewardIndex(IntEnum):
    MOTION_TRACKING = 0
    LIN_VEL_TRACKING = 1
    YAW_VEL_TRACKING = 2
    TARGET_HIT = 3


NUM_REWARDS = len(RewardIndex)
