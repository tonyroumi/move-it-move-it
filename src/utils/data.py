from typing import Any, Union
import numpy as np
import torch

ArrayLike = Union[np.ndarray, torch.Tensor]

def _is_tensor(x: Any) -> bool:
    return isinstance(x, torch.Tensor)

def _to_numpy(x: Any) -> np.ndarray:
    if _is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

def _to_torch(x: Any, device: torch.device = 'cpu') -> torch.Tensor:
    if _is_tensor(x):
        return x.to(device)
    return torch.tensor(x, dtype=torch.float32, device=device)