from dataclasses import dataclass

from moveitmoveit.agents.ppo import PPOCfg


@dataclass(kw_only=True)
class AMPCfg(PPOCfg):
    # Discriminator replay-buffer settings
    discriminator_buffer_capacity: int = 100_000
    disc_replay_samples: int = 1000
    # Discriminator optimizer
    disc_lr: float = 1e-4
    disc_logit_reg: float = 0.01
    disc_weight_decay: float = 0.0001

    # How often (in policy update iterations) to update the discriminator
    discriminator_update_interval: int = 4

    # Number of gradient steps per discriminator update
    disc_num_updates: int = 5
    disc_batch_size: int = 2048

    disc_grad_penalty_coef: float = 5.0
    num_disc_obs_steps: int = 10

    disc_reward_lambda: float = 2 
    goal_reward_lambda: float = 1.0
