from dataclasses import dataclass

from moveitmoveit.agents.base import BaseCfg


@dataclass(kw_only=True)
class PPOCfg(BaseCfg):

    num_mini_batches: int = 1
    """Number of mini-batches used to split each collected rollout batch."""

    num_learning_epochs: int = 4
    """Number of optimization passes over the collected rollout data."""

    learning_rate: float = 3e-4
    """Learning rate used by the optimizer."""

    max_grad_norm: float = 1.0
    """Maximum gradient norm used for gradient clipping."""

    clip_param: float = 0.2
    """PPO clipping parameter ε used to constrain policy updates. """

    value_loss_clip_param: float = 1.0
    """Value loss clipped parameter used to constrain critic updates. """

    value_loss_coef: float = 1.0
    """Coefficient applied to the critic/value-function loss."""

    entropy_coef: float = 0.2
    """Coefficient applied to the policy entropy bonus."""

    use_clipped_value_loss: bool = True
    """Whether to apply PPO-style clipping to value-function updates."""

    adaptive_kl: bool = False
    """Whether to use kl penalty objective vs clipped objective."""

    desired_kl: float = 0.01
    """Target KL divergence between the old and updated policies."""

    discount: float = 0.97
    """Reward discount factor γ used when computing discounted returns."""

    gae_lambda: float = 0.95
    """GAE parameter λ controlling the bias-variance tradeoff."""

    normalize_advantage_per_mini_batch: bool = False
    """Whether to normalize advantages independently within each mini-batch.

    If False, advantages should typically be normalized over the full rollout
    batch before it is divided into mini-batches.
    """