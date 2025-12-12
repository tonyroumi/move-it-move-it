from .data import ArrayLike, ArrayUtils

from scipy.spatial.transform import Rotation as R
import torch

class RotationUtils:
    """Rotation representation conversion utilities."""

    @staticmethod
    def aa_to_quat(
        axis_angles: ArrayLike,
        return_torch: bool = False,
        device: torch.device = None,
    ):
        """Convert axis-angle vectors to quaternions [x, y, z, w]."""
        out = R.from_rotvec(ArrayUtils.to_numpy(axis_angles)).as_quat()
        return ArrayUtils.to_torch(out, device) if return_torch else out

    @staticmethod
    def aa_to_rotmat(
        axis_angles: ArrayLike,
        return_torch: bool = False,
        device: torch.device = None,
    ):
        """Convert axis-angle vectors to rotation matrices."""
        out = R.from_rotvec(ArrayUtils.to_numpy(axis_angles)).as_matrix()
        return ArrayUtils.to_torch(out, device) if return_torch else out

    @staticmethod
    def matrix_to_quat(
        matrices: ArrayLike,
        return_torch: bool = False,
        device: torch.device = None,
    ):
        """Convert rotation matrices to quaternions [x, y, z, w]."""
        out = R.from_matrix(ArrayUtils.to_numpy(matrices)).as_quat()
        return ArrayUtils.to_torch(out, device) if return_torch else out