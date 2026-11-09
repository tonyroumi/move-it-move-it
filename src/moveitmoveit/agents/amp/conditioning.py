"""Conditional AMP: which generated samples carry one behavior condition, and the discriminator input."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def discriminator_input(observations: torch.Tensor, conditions: torch.Tensor | None, num_conditions: int) -> torch.Tensor:
    """Normalized AMP observations, followed by a one-hot of each sample's condition when the
    discriminator is conditional (`num_conditions` > 0). The one-hot is never normalized."""
    if not num_conditions:
        return observations
    one_hot = F.one_hot(conditions.long(), num_conditions).to(observations.dtype)
    return torch.cat((observations, one_hot), dim=-1)


class StyleConditionTracker:
    """Per env, whether this step's AMP observation is a policy-generated sample of one condition.

    The env's AMP observation after a step holds `num_amp_steps` frames, newest first, joined by the
    last `num_amp_steps - 1` policy transitions. It is a sample of condition c only when every one of
    those transitions was taken under c. So it is not one when:
    - the env was reset this step: its history is the new episode's reference seed, not policy motion;
    - fewer than `num_amp_steps - 1` steps have passed since a reset or a condition change, so the
      history still holds seed frames or motion made under another condition (with 2 frames, the
      first transition after a change is already a clean sample of the new condition);
    - the condition is -1 (standing), which belongs to no condition.
    """

    def __init__(self, num_envs: int, num_amp_steps: int, device: torch.device | str):
        self._required = max(num_amp_steps - 1, 1)
        # the condition of the last transition, and how many consecutive transitions it has lasted
        self._condition = torch.full((num_envs,), -1, dtype=torch.long, device=device)
        self._run_length = torch.zeros(num_envs, dtype=torch.long, device=device)

    def update(self, conditions: torch.Tensor, resets: torch.Tensor) -> torch.Tensor:
        """Record this step's transitions, taken under `conditions` (N,), with `resets` (N,) the envs
        whose episode ended this step. Returns the (N,) bool mask of usable samples."""
        same = conditions == self._condition
        self._run_length = torch.where(same, self._run_length + 1, torch.ones_like(self._run_length))
        self._condition = conditions.clone()

        valid = (self._run_length >= self._required) & (conditions >= 0) & ~resets

        # the reset envs' history now starts over from the reference seed
        self._run_length[resets] = 0
        self._condition[resets] = -1
        return valid
