from enum import IntEnum


class RewardIndex(IntEnum):
    MOTION_TRACKING = 0
    LIN_VEL_TRACKING = 1
    YAW_VEL_TRACKING = 2
    TARGET_HIT = 3
    LINE_FOLLOWING = 4
    ACTION_RATE_L2 = 5
    JOINT_ACC = 6
    JOINT_VEL = 7
    PROGRESS = 8


NUM_REWARDS = len(RewardIndex)


REWARD_KWARG_DEFAULTS: dict[str, dict[str, float]] = {
    "motion_tracking": {"dof_scale": 2.0, "key_body_scale": 5.0},
    "lin_vel_tracking": {"scale": 2.0},
    "yaw_vel_tracking": {"scale": 2.0},
    "target_hit": {"scale": 0.5},
    "line_following": {"scale": 2.0},
    "action_rate_l2": {"scale": 1.0},
    "joint_acc": {"scale": 1.0e-4},
    "joint_vel": {"scale": 1.0e-2},
    "progress": {"scale": 1.0},
}


class TerminationIndex(IntEnum):
    DEVIATION_FROM_MOTION = 0
    UNHEALTHY = 1


NUM_TERMINATIONS = len(TerminationIndex)
