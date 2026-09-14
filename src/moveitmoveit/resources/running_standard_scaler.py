from __future__ import annotations

import torch
import torch.nn as nn


class RunningStandardScaler(nn.Module):
    """Normalizes values using running mean/variance statistics, updated batch-by-batch
    via Chan et al.'s parallel variance algorithm for numerically stable combination of
    running and batch statistics.

    The last `unscaled_dims` elements of the final dimension are left untouched (never
    tracked, never normalized) and passed through as-is, for values that are already in
    a known range (e.g. one-hot ids, pre-scaled commands) and shouldn't be normalized.

    `mean`/`variance` seed the running statistics at construction (broadcast to the
    scaled shape) instead of the zeros/ones default. Combined with never calling
    `forward(..., train=True)`, this freezes the scaler at a known fixed mean/variance
    rather than one learned from data."""

    def __init__(
        self,
        size: int | tuple[int, ...],
        epsilon: float = 1e-8,
        clip_threshold: float = 5.0,
        unscaled_dims: int = 0,
        mean: torch.Tensor | float | None = None,
        variance: torch.Tensor | float | None = None,
    ):
        super().__init__()
        self.epsilon = epsilon
        self.clip_threshold = clip_threshold

        shape = (size,) if isinstance(size, int) else tuple(size)
        if not (0 <= unscaled_dims <= shape[-1]):
            raise ValueError(f"unscaled_dims ({unscaled_dims}) must be within [0, {shape[-1]}]")
        self.unscaled_dims = unscaled_dims

        scaled_shape = shape[:-1] + (shape[-1] - unscaled_dims,)

        initial_mean = torch.zeros(scaled_shape, dtype=torch.float64) if mean is None \
            else torch.as_tensor(mean, dtype=torch.float64).expand(scaled_shape)
        initial_variance = torch.ones(scaled_shape, dtype=torch.float64) if variance is None \
            else torch.as_tensor(variance, dtype=torch.float64).expand(scaled_shape)

        self.register_buffer("running_mean", initial_mean.clone())
        self.register_buffer("running_variance", initial_variance.clone())
        self.register_buffer("count", torch.tensor(1e-4, dtype=torch.float64))

    def _update(self, x: torch.Tensor) -> None:
        # reduce over every dim but the last, so a (batch, ..., features) input is treated
        # as a flat batch of feature vectors regardless of how many leading dims it has
        reduce_dims = tuple(range(x.dim() - 1))
        batch_mean = torch.mean(x.double(), dim=reduce_dims)
        batch_variance = torch.var(x.double(), dim=reduce_dims, unbiased=False)
        batch_count = x.shape[:-1].numel()

        delta = batch_mean - self.running_mean
        total_count = self.count + batch_count

        new_mean = self.running_mean + delta * batch_count / total_count
        m_a = self.running_variance * self.count
        m_b = batch_variance * batch_count
        new_variance = (m_a + m_b + delta**2 * self.count * batch_count / total_count) / total_count

        self.running_mean.copy_(new_mean)
        self.running_variance.copy_(new_variance)
        self.count.copy_(total_count)

    @torch.no_grad()
    def forward(self, x: torch.Tensor, train: bool = False, inverse: bool = False) -> torch.Tensor:
        if self.unscaled_dims:
            x, unscaled = x[..., : -self.unscaled_dims], x[..., -self.unscaled_dims :]
        else:
            unscaled = None

        if train:
            self._update(x)

        mean = self.running_mean.to(x.dtype)
        std = torch.sqrt(self.running_variance).to(x.dtype) + self.epsilon

        if inverse:
            result = x * std + mean
        else:
            result = torch.clamp((x - mean) / std, min=-self.clip_threshold, max=self.clip_threshold)

        return result if unscaled is None else torch.cat((result, unscaled), dim=-1)
