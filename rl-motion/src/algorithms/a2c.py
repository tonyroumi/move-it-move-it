"""Advantage Actor-Critic (A2C) algorithm."""

import torch
from typing import Dict, Tuple

from motion.src.core.types import RolloutBatch, PolicyOutput
from motion.src.core.hyperparams import A2CHyperparams
from motion.src.components.advantage import Advantage
from motion.src.components.policy_loss import PolicyLoss
from motion.src.components.value_loss import ValueLoss


class A2C:
    """A2C (Advantage Actor-Critic) implemented as static methods.

    Key differences from PPO:
    - No clipping on policy loss (uses vanilla policy gradient)
    - Single gradient step per rollout (no multiple epochs)
    - Can use n-step returns or GAE for advantage estimation

    Example usage:
        # Compute advantages after rollout
        batch = A2C.compute_advantages(batch, next_value, hyperparams)

        # Single gradient step
        loss, metrics = A2C.compute_loss(
            policy_output, batch.advantages, batch.returns, hyperparams
        )
        loss.backward()
        optimizer.step()
    """

    @staticmethod
    def compute_advantages(
        batch: RolloutBatch,
        next_value: torch.Tensor,
        hyperparams: A2CHyperparams,
    ) -> RolloutBatch:
        """Compute advantages and returns for a rollout batch.

        Uses either GAE or simple n-step returns based on hyperparameters.

        Args:
            batch: Rollout batch without advantages/returns computed.
            next_value: Bootstrap value for final state [B] or scalar.
            hyperparams: A2C hyperparameters.

        Returns:
            New RolloutBatch with advantages and returns filled in.
        """
        if hyperparams.use_gae:
            advantages, returns = Advantage.gae(
                rewards=batch.rewards,
                values=batch.values,
                dones=batch.dones,
                next_value=next_value,
                gamma=hyperparams.gamma,
                gae_lambda=hyperparams.gae_lambda,
            )
        else:
            returns = Advantage.n_step_returns(
                rewards=batch.rewards,
                dones=batch.dones,
                next_value=next_value,
                gamma=hyperparams.gamma,
            )
            advantages = returns - batch.values

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
        advantages: torch.Tensor,
        returns: torch.Tensor,
        hyperparams: A2CHyperparams,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute A2C loss.

        Uses vanilla policy gradient (no clipping).

        Args:
            policy_output: Output from actor network.
            advantages: Computed advantages [B].
            returns: Computed returns [B].
            hyperparams: A2C hyperparameters.

        Returns:
            total_loss: Combined loss for backpropagation.
            metrics: Dictionary of loss components.
        """
        policy_loss = PolicyLoss.vanilla_pg(
            log_probs=policy_output.log_prob,
            advantages=advantages,
        )

        if policy_output.value is not None:
            values = policy_output.value
        else:
            raise ValueError("A2C requires value estimates in PolicyOutput")

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
        }

        return total_loss, metrics

    @staticmethod
    def compute_loss_from_batch(
        actor: torch.nn.Module,
        critic: torch.nn.Module,
        batch: RolloutBatch,
        hyperparams: A2CHyperparams,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute A2C loss directly from batch and networks.

        Convenience method that handles network forward passes.

        Args:
            actor: Actor network (must have evaluate_actions method).
            critic: Critic network.
            batch: Rollout batch with computed advantages.
            hyperparams: A2C hyperparameters.

        Returns:
            total_loss: Combined loss.
            metrics: Loss metrics.
        """
        flat_batch = batch.flatten()

        log_probs, entropy = actor.evaluate_actions(
            flat_batch.observations, flat_batch.actions
        )
        values = critic.get_value(flat_batch.observations)

        policy_output = PolicyOutput(
            action_mean=torch.zeros_like(flat_batch.actions),
            action_std=torch.ones_like(flat_batch.actions),
            log_prob=log_probs,
            entropy=entropy,
            value=values,
        )

        return A2C.compute_loss(
            policy_output=policy_output,
            advantages=flat_batch.advantages,
            returns=flat_batch.returns,
            hyperparams=hyperparams,
        )
