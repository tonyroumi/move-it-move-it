"""Neural network modules."""

from motion.src.networks.actor import GaussianActor, ActorConfig
from motion.src.networks.critic import ValueCritic, CriticConfig
from motion.src.networks.discriminator import MotionDiscriminator, DiscriminatorConfig

__all__ = [
    "GaussianActor",
    "ActorConfig",
    "ValueCritic",
    "CriticConfig",
    "MotionDiscriminator",
    "DiscriminatorConfig",
]
