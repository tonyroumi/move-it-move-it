"""Adversarial Motion Prior (AMP) algorithm."""

import torch
from typing import Dict, Tuple, Optional

from motion.src.core.types import RolloutBatch, MotionState, PolicyOutput
from motion.src.core.hyperparams import AMPHyperparams
from motion.src.algorithms.ppo import PPO
from motion.src.networks.discriminator import MotionDiscriminator
from motion.src.rewards.discriminator import DiscriminatorReward, DiscriminatorRewardConfig


class AMP:
    """Adversarial Motion Prior (AMP) implemented as static methods.

    AMP extends PPO with a learned discriminator that provides style rewards.
    The discriminator learns to distinguish reference motion from policy-generated
    motion, and rewards the policy for producing motion that looks "real".

    The total reward is a weighted combination of task reward and style reward:
        reward = task_weight * task_reward + style_weight * style_reward

    Example usage:
        # Compute style rewards from discriminator
        style_rewards = AMP.compute_style_reward(discriminator, motion_states)

        # Blend with task rewards
        blended_rewards = AMP.blend_rewards(
            task_rewards, style_rewards, hyperparams
        )

        # Use PPO for policy update
        batch = PPO.compute_advantages(batch, next_value, hyperparams.ppo)
        loss, metrics = PPO.compute_loss_from_batch(actor, critic, batch, hyperparams.ppo)

        # Update discriminator separately
        disc_loss, disc_metrics = AMP.compute_discriminator_loss(
            discriminator, real_states, fake_states, hyperparams
        )
    """

    @staticmethod
    def compute_style_reward(
        discriminator: MotionDiscriminator,
        motion_states: MotionState,
    ) -> torch.Tensor:
        """Compute style reward from discriminator.

        Args:
            discriminator: Trained motion discriminator.
            motion_states: Motion states to evaluate.

        Returns:
            Style rewards [B].
        """
        motion_features = motion_states.flatten_features()

        with torch.no_grad():
            reward = discriminator.get_reward(motion_features)

        return reward

    @staticmethod
    def blend_rewards(
        task_rewards: torch.Tensor,
        style_rewards: torch.Tensor,
        hyperparams: AMPHyperparams,
    ) -> torch.Tensor:
        """Blend task and style rewards.

        Args:
            task_rewards: Task-specific rewards [B] or [B, T].
            style_rewards: Style rewards from discriminator [B] or [B, T].
            hyperparams: AMP hyperparameters.

        Returns:
            Blended rewards.
        """
        return (
            hyperparams.task_reward_weight * task_rewards
            + hyperparams.style_reward_weight * style_rewards
        )

    @staticmethod
    def compute_discriminator_loss(
        discriminator: MotionDiscriminator,
        real_states: MotionState,
        fake_states: MotionState,
        hyperparams: AMPHyperparams,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute discriminator training loss.

        Args:
            discriminator: Motion discriminator network.
            real_states: Motion states from reference dataset.
            fake_states: Motion states from policy rollout.
            hyperparams: AMP hyperparameters.

        Returns:
            loss: Discriminator loss.
            metrics: Loss components.
        """
        config = DiscriminatorRewardConfig(
            gradient_penalty_coef=hyperparams.discriminator.gradient_penalty_coef,
        )

        disc_reward = DiscriminatorReward(discriminator, config)
        loss, metrics = disc_reward.compute_discriminator_loss(real_states, fake_states)

        return loss, {k: v.item() if torch.is_tensor(v) else v for k, v in metrics.items()}

    @staticmethod
    def compute_policy_loss(
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        batch: RolloutBatch,
        hyperparams: AMPHyperparams,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute policy loss using PPO objective.

        AMP uses PPO as the underlying policy optimization algorithm.

        Args:
            actor: Actor network.
            critic: Critic network.
            batch: Rollout batch with blended rewards and computed advantages.
            hyperparams: AMP hyperparameters.

        Returns:
            loss: Policy loss.
            metrics: Loss metrics.
        """
        return PPO.compute_loss_from_batch(actor, critic, batch, hyperparams.ppo)

    @staticmethod
    def compute_advantages(
        batch: RolloutBatch,
        next_value: torch.Tensor,
        hyperparams: AMPHyperparams,
    ) -> RolloutBatch:
        """Compute advantages using PPO's GAE.

        Args:
            batch: Rollout batch (should have blended rewards).
            next_value: Bootstrap value.
            hyperparams: AMP hyperparameters.

        Returns:
            Batch with computed advantages.
        """
        return PPO.compute_advantages(batch, next_value, hyperparams.ppo)

    @staticmethod
    def update_rewards_in_batch(
        batch: RolloutBatch,
        discriminator: MotionDiscriminator,
        motion_states: MotionState,
        hyperparams: AMPHyperparams,
    ) -> RolloutBatch:
        """Update batch with blended task and style rewards.

        Assumes batch.rewards contains task rewards and computes style rewards
        from the discriminator to create blended rewards.

        Args:
            batch: Rollout batch with task rewards.
            discriminator: Motion discriminator.
            motion_states: Motion states corresponding to batch.
            hyperparams: AMP hyperparameters.

        Returns:
            New batch with blended rewards.
        """
        style_rewards = AMP.compute_style_reward(discriminator, motion_states)

        if batch.rewards.dim() > style_rewards.dim():
            style_rewards = style_rewards.unsqueeze(-1).expand_as(batch.rewards)

        blended_rewards = AMP.blend_rewards(batch.rewards, style_rewards, hyperparams)

        return RolloutBatch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=blended_rewards,
            dones=batch.dones,
            log_probs=batch.log_probs,
            values=batch.values,
            advantages=batch.advantages,
            returns=batch.returns,
        )

    @staticmethod
    def get_minibatches(
        batch: RolloutBatch,
        hyperparams: AMPHyperparams,
        shuffle: bool = True,
    ):
        """Generate minibatches using PPO's minibatch generator.

        Args:
            batch: Full rollout batch.
            hyperparams: AMP hyperparameters.
            shuffle: Whether to shuffle indices.

        Yields:
            Minibatch RolloutBatch instances.
        """
        yield from PPO.get_minibatches(
            batch, hyperparams.ppo.num_minibatches, shuffle
        )
