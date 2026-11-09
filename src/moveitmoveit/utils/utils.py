import torch


def explained_variance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Explained variance of `predictions` w.r.t. `targets`: 1 is perfect, 0 no better than the
    target mean, below 0 worse; NaN when the targets barely vary."""
    target_var = torch.var(targets, unbiased=False)

    if target_var < eps:
        return torch.tensor(float("nan"), device=targets.device)

    residual_var = torch.var(targets - predictions, unbiased=False)

    return 1.0 - residual_var / target_var


def fraction_outside_bounds(
    values: torch.Tensor,
    lower: float,
    upper: float,
) -> torch.Tensor:
    """Return the fraction of values outside [lower, upper]."""
    return ((values < lower) | (values > upper)).float().mean()
