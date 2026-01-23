"""Example usage of the RL algorithm framework.

This module demonstrates:
1. Using PPO, A2C, and AMP independently
2. Sharing components between algorithms
3. Combining reward modules
4. Setting up training loops
"""

import torch
import torch.nn as nn
from typing import Tuple

from motion.src.core.types import RolloutBatch, MotionState, PolicyOutput
from motion.src.core.hyperparams import (
    PPOHyperparams,
    A2CHyperparams,
    AMPHyperparams,
    DiscriminatorHyperparams,
)
from motion.src.algorithms.ppo import PPO
from motion.src.algorithms.a2c import A2C
from motion.src.algorithms.amp import AMP
from motion.src.components.advantage import Advantage
from motion.src.components.policy_loss import PolicyLoss
from motion.src.networks.actor import GaussianActor, ActorConfig
from motion.src.networks.critic import ValueCritic, CriticConfig
from motion.src.networks.discriminator import MotionDiscriminator, DiscriminatorConfig
from motion.src.rewards.motion_matching import MotionMatchingReward, MotionMatchingConfig
from motion.src.rewards.discriminator import DiscriminatorReward, DiscriminatorRewardConfig
from motion.src.rewards.composite import CompositeReward


# =============================================================================
# Example 1: Basic PPO Training Loop
# =============================================================================

def example_ppo_training():
    """Demonstrates a basic PPO training setup."""
    print("=" * 60)
    print("Example 1: PPO Training")
    print("=" * 60)

    # Configuration
    obs_dim = 64
    action_dim = 16
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Hyperparameters as frozen dataclass
    hyperparams = PPOHyperparams(
        clip_epsilon=0.2,
        learning_rate=3e-4,
        num_epochs=10,
        num_minibatches=4,
        gamma=0.99,
        gae_lambda=0.95,
        value_loss_coef=0.5,
        entropy_coef=0.01,
    )

    # Create networks with their own configs
    actor_config = ActorConfig(
        input_dim=obs_dim,
        output_dim=action_dim,
        hidden_dims=(256, 256),
        activation="elu",
    )
    critic_config = CriticConfig(
        input_dim=obs_dim,
        hidden_dims=(256, 256),
    )

    actor = GaussianActor(actor_config).to(device)
    critic = ValueCritic(critic_config).to(device)

    optimizer = torch.optim.Adam(
        list(actor.parameters()) + list(critic.parameters()),
        lr=hyperparams.learning_rate,
    )

    # Simulate a rollout batch
    batch_size, seq_len = 32, 64
    batch = _create_dummy_batch(batch_size, seq_len, obs_dim, action_dim, device)

    # Compute next value for bootstrapping
    with torch.no_grad():
        next_obs = torch.randn(batch_size, obs_dim, device=device)
        next_value = critic.get_value(next_obs)

    # PPO advantage computation (static method)
    batch = PPO.compute_advantages(batch, next_value, hyperparams)

    print(f"Advantages shape: {batch.advantages.shape}")
    print(f"Returns shape: {batch.returns.shape}")

    # Training loop using static methods
    for epoch in range(hyperparams.num_epochs):
        epoch_metrics = []

        for minibatch in PPO.get_minibatches(batch, hyperparams.num_minibatches):
            loss, metrics = PPO.compute_loss_from_batch(
                actor, critic, minibatch, hyperparams
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(actor.parameters()) + list(critic.parameters()),
                hyperparams.max_grad_norm,
            )
            optimizer.step()

            epoch_metrics.append(metrics)

        avg_loss = sum(m["total_loss"] for m in epoch_metrics) / len(epoch_metrics)
        print(f"  Epoch {epoch + 1}: loss={avg_loss:.4f}")

    print("PPO training complete!\n")


# =============================================================================
# Helper Functions
# =============================================================================

def _create_dummy_batch(
    batch_size: int,
    seq_len: int,
    obs_dim: int,
    action_dim: int,
    device: torch.device,
) -> RolloutBatch:
    """Create a dummy rollout batch for testing."""
    return RolloutBatch(
        observations=torch.randn(batch_size, seq_len, obs_dim, device=device),
        actions=torch.randn(batch_size, seq_len, action_dim, device=device),
        rewards=torch.rand(batch_size, seq_len, device=device),
        dones=torch.zeros(batch_size, seq_len, device=device),
        log_probs=torch.randn(batch_size, seq_len, device=device),
        values=torch.randn(batch_size, seq_len, device=device),
        advantages=torch.zeros(batch_size, seq_len, device=device),
        returns=torch.zeros(batch_size, seq_len, device=device),
    )


def _create_dummy_motion_state(
    batch_size: int,
    num_joints: int,
    device: torch.device,
) -> MotionState:
    """Create a dummy motion state for testing."""
    return MotionState(
        joint_positions=torch.randn(batch_size, num_joints, 3, device=device),
        joint_rotations=torch.randn(batch_size, num_joints, 4, device=device),
        joint_velocities=torch.randn(batch_size, num_joints, 3, device=device),
        root_position=torch.randn(batch_size, 3, device=device),
        root_velocity=torch.randn(batch_size, 3, device=device),
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Motion RL Framework - Algorithm Usage Examples")
    print("=" * 60 + "\n")

    example_ppo_training()

    print("=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
