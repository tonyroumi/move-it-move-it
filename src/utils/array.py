from typing import Any, Union
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

class ArrayUtils:
    """Array and tensor conversion utilities."""

    @staticmethod
    def is_tensor(x: Any) -> bool:
        """Return True if input is a torch Tensor."""
        return isinstance(x, torch.Tensor)

    @staticmethod
    def to_numpy(x: Any) -> np.ndarray:
        """Convert input to a NumPy array."""
        if ArrayUtils.is_tensor(x):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    @staticmethod
    def to_torch(x: Any, device: torch.device = "cpu") -> torch.Tensor:
        """Convert input to a torch Tensor."""
        if ArrayUtils.is_tensor(x):
            return x.to(device)
        return torch.tensor(x, dtype=torch.float32, device=device)