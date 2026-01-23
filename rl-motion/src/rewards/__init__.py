"""Reward computation modules."""

from motion.src.rewards.base import RewardModule
from motion.src.rewards.motion_matching import MotionMatchingReward, MotionMatchingConfig
from motion.src.rewards.discriminator import DiscriminatorReward, DiscriminatorRewardConfig
from motion.src.rewards.composite import CompositeReward

__all__ = [
    "RewardModule",
    "MotionMatchingReward",
    "MotionMatchingConfig",
    "DiscriminatorReward",
    "DiscriminatorRewardConfig",
    "CompositeReward",
]
