from .data import ArrayLike, _is_tensor

import numpy as np
import torch

def prune_joints(data: ArrayLike, cutoff: int, exclude_root: bool = True) -> ArrayLike:
    """
    Prune joints based on an index cutoff, optionally excluding the root joint.

    Args:
        data: (J, ...) or (T, J, ...) containing indexed joint information. 

    Returns:
        Pruned joint data.

    NOTE: assumes that joint 0 is the root joint. 
    """
    if _is_tensor(data):
        all_idx = torch.arange(cutoff, device=data.device)
    else:
        all_idx = np.arange(cutoff)
      
    idx = all_idx if not exclude_root else all_idx[1:]

    # Joint-major: [J, ...]
    if data.ndim == 2:
        return data[idx]
    # Batch-major: [N, J, ...]
    elif data.ndim == 3:
        return data[:, idx]
