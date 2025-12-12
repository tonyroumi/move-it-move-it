from .data import ArrayLike, ArrayUtils

import numpy as np
import torch

class JointUtils:
    """Joint indexing utilities."""

    @staticmethod
    def prune_joints(data: ArrayLike, cutoff: int) -> ArrayLike:
        """ Prune joints based on an index cutoff. """
        if ArrayUtils.is_tensor(data):
            all_idx = torch.arange(cutoff, device=data.device)
        else:
            all_idx = np.arange(cutoff)

        # Joint-major: (J, ...)
        if data.ndim == 2:
            return data[all_idx]

        # Batch-major: (N, J, ...)
        elif data.ndim == 3:
            return data[:, all_idx]
