import torch


def explained_variance(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute the explained variance of predictions with respect to targets.

    Returns:
        1.0: perfect predictions
        0.0: no better than predicting the target mean
        <0.0: worse than predicting the target mean
    """
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