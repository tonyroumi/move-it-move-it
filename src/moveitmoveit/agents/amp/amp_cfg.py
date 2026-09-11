from dataclasses import dataclass

from moveitmoveit.agents.ppo import PPOCfg


@dataclass(kw_only=True)
class AMPCfg(PPOCfg):

    discriminator_buffer_capacity: int = 100_000
    """Maximum number of discriminator transitions stored in the replay buffer."""

    disc_lr: float = 1e-4
    """Learning rate used by the discriminator optimizer."""

    disc_logit_reg: float = 0.01
    """Coefficient applied to discriminator logit regularization."""

    disc_grad_penalty: float = 5.0
    """Coefficient applied to the discriminator gradient penalty."""

    disc_weight_decay: float = 0.0001
    """Weight decay applied by the discriminator optimizer."""

    disc_loss_scale: float = 1.0
    """Global scaling factor applied to the discriminator loss."""

    grad_norm_clip: float = 1.0
    """Maximum discriminator gradient norm used for gradient clipping."""

    discriminator_update_interval: int = 1
    """Number of policy update iterations between discriminator updates."""

    disc_num_updates: int = 5
    """Number of discriminator optimization steps performed per discriminator update."""

    disc_batch_size: int = 2048
    """Batch size used for each discriminator optimization step."""

    style_reward_lambda: float = 2.0
    """Coefficient applied to the AMP style/imitation reward."""

    goal_reward_lambda: float = 1.0
    """Coefficient applied to the task or goal reward."""

    num_amp_observations: int = 2
    """Number of consecutive observation frames stacked into one AMP/style observation."""
