from dataclasses import dataclass

from moveitmoveit.agents.base import BaseCfg


@dataclass(kw_only=True)
class PPOCfg(BaseCfg):
    num_mini_batches: int = 1
    num_learning_epochs: int = 4
    clip_param: float = 0.2
    discount: float = 0.97
    td_lambda: float = 0.95
    lr: float = 3e-4
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    desired_kl: float = 0.01
    normalize_advantage_per_mini_batch: bool = False

    value_loss_coef: float = 1.0
    entropy_coef: float = 0.2
