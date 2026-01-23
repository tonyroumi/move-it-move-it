"""Proximal Policy Optimization (PPO) algorithm."""

import torch
from typing import Dict, Tuple

from motion.src.core.types import RolloutBatch, PolicyOutput
from motion.src.core.hyperparams import PPOHyperparams
from motion.src.components.advantage import Advantage
from motion.src.components.policy_loss import PolicyLoss
from motion.src.components.value_loss import ValueLoss


class PPO:
    """PPO algorithm implemented as static methods.

    All state (networks, optimizers) lives outside this class.
    This class provides pure functional transformations for PPO training.

    Example usage:
        # Compute advantages after rollout
        batch = PPO.compute_advantages(batch, next_value, hyperparams)

        # Training loop
        for epoch in range(hyperparams.num_epochs):
            for minibatch in get_minibatches(batch):
                loss, metrics = PPO.compute_loss(
                    policy_output, minibatch.log_probs, minibatch.values,
                    minibatch.advantages, minibatch.returns, hyperparams
                )
                loss.backward()
                optimizer.step()
    """

    @staticmethod
    def compute_advantages(
        batch: RolloutBatch,
        next_value: torch.Tensor,
        hyperparams: PPOHyperparams,
    ) -> RolloutBatch:
        """Compute advantages and returns for a rollout batch.

        Args:
            batch: Rollout batch without advantages/returns computed.
            next_value: Bootstrap value for final state [B] or scalar.
            hyperparams: PPO hyperparameters.

        Returns:
            New RolloutBatch with advantages and returns filled in.
        """
        advantages, returns = Advantage.gae(
            rewards=batch.rewards,
            values=batch.values,
            dones=batch.dones,
            next_value=next_value,
            gamma=hyperparams.gamma,
            gae_lambda=hyperparams.gae_lambda,
        )

        if hyperparams.normalize_advantage:
            advantages = Advantage.normalize(advantages)

        return RolloutBatch(
            observations=batch.observations,
            actions=batch.actions,
            rewards=batch.rewards,
            dones=batch.dones,
            log_probs=batch.log_probs,
            values=batch.values,
            advantages=advantages,
            returns=returns,
        )

    @staticmethod
    def compute_loss(
        policy_output: PolicyOutput,
        old_log_probs: torch.Tensor,
        old_values: torch.Tensor,
        advantages: torch.Tensor,
        returns: torch.Tensor,
        hyperparams: PPOHyperparams,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss.

        Args:
            policy_output: Output from actor network containing log_prob, entropy, value.
            old_log_probs: Log probabilities from rollout collection [B].
            old_values: Value estimates from rollout collection [B].
            advantages: Computed advantages [B].
            returns: Computed returns [B].
            hyperparams: PPO hyperparameters.

        Returns:
            total_loss: Combined loss for backpropagation.
            metrics: Dictionary of loss components and diagnostics.
        """
        policy_loss, policy_metrics = PolicyLoss.clipped_surrogate(
            log_probs=policy_output.log_prob,
            old_log_probs=old_log_probs,
            advantages=advantages,
            clip_epsilon=hyperparams.clip_epsilon,
        )

        if policy_output.value is not None:
            values = policy_output.value
        else:
            raise ValueError("PPO requires value estimates in PolicyOutput")

        if hyperparams.clip_value_loss:
            value_loss = ValueLoss.clipped(
                values=values,
                old_values=old_values,
                returns=returns,
                clip_epsilon=hyperparams.value_clip_epsilon,
            )
        else:
            value_loss = ValueLoss.mse(values, returns)

        entropy_loss = -policy_output.entropy.mean()

        total_loss = (
            policy_loss
            + hyperparams.value_loss_coef * value_loss
            + hyperparams.entropy_coef * entropy_loss
        )

        metrics = {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": -entropy_loss.item(),
            "total_loss": total_loss.item(),
            "clip_fraction": policy_metrics["clip_fraction"].item(),
            "approx_kl": policy_metrics["approx_kl"].item(),
        }

        return total_loss, metrics

    @staticmethod
    def compute_loss_from_batch(
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        batch: RolloutBatch,
        hyperparams: PPOHyperparams,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute PPO loss directly from batch and networks.

        Convenience method that handles network forward passes.

        Args:
            actor: Actor network (must have evaluate_actions method).
            critic: Critic network.
            batch: Rollout batch with computed advantages.
            hyperparams: PPO hyperparameters.

        Returns:
            total_loss: Combined loss.
            metrics: Loss metrics.
        """
        log_probs, entropy = actor.evaluate_actions(batch.observations, batch.actions)
        values = critic.get_value(batch.observations)

        policy_output = PolicyOutput(
            action_mean=torch.zeros_like(batch.actions),
            action_std=torch.ones_like(batch.actions),
            log_prob=log_probs,
            entropy=entropy,
            value=values,
        )

        return PPO.compute_loss(
            policy_output=policy_output,
            old_log_probs=batch.log_probs,
            old_values=batch.values,
            advantages=batch.advantages,
            returns=batch.returns,
            hyperparams=hyperparams,
        )

    @staticmethod
    def get_minibatches(
        batch: RolloutBatch,
        num_minibatches: int,
        shuffle: bool = True,
    ):
        """Generate minibatches from a rollout batch.

        Args:
            batch: Full rollout batch.
            num_minibatches: Number of minibatches to create.
            shuffle: Whether to shuffle indices.

        Yields:
            Minibatch RolloutBatch instances.
        """
        flat_batch = batch.flatten()
        batch_size = flat_batch.observations.shape[0]
        minibatch_size = batch_size // num_minibatches

        if shuffle:
            indices = torch.randperm(batch_size, device=flat_batch.observations.device)
        else:
            indices = torch.arange(batch_size, device=flat_batch.observations.device)

        for start in range(0, batch_size, minibatch_size):
            end = start + minibatch_size
            mb_indices = indices[start:end]

            yield RolloutBatch(
                observations=flat_batch.observations[mb_indices],
                actions=flat_batch.actions[mb_indices],
                rewards=flat_batch.rewards[mb_indices],
                dones=flat_batch.dones[mb_indices],
                log_probs=flat_batch.log_probs[mb_indices],
                values=flat_batch.values[mb_indices],
                advantages=flat_batch.advantages[mb_indices],
                returns=flat_batch.returns[mb_indices],
            )
