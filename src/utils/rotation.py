from .data import ArrayLike, _to_torch, _to_numpy

from scipy.spatial.transform import Rotation as R
import torch

def axis_angle_to_quat(axis_angles: ArrayLike, return_torch: bool = False, device: torch.device = None):
    """
    Convert axis-angle representations to quaternions.
    
    Args:
        axis_angles: (N, 3) or (3,) axis-angle vector(s)
    
    Returns:
        quats: (N, 4) or (4,) quaternions [x, y, z, w].        
    """
    out = R.from_rotvec(_to_numpy(axis_angles)).as_quat()
    return _to_torch(out, device) if return_torch else out


def axis_angle_to_matrix(axis_angles: ArrayLike, return_torch: bool = False, device: torch.device = None):
    """
    Convert axis-angle representations to rotation matrices.
    
    Args:
        axis_angles: (N, 3) or (3,) containing axis-angle vectors.
    
    Returns:
        matrices: (N, 3, 3) or (3, 3) rotation matrices. 
    """
    out = R.from_rotvec(_to_numpy(axis_angles)).as_matrix()
    return _to_torch(out, device) if return_torch else out


def matrix_to_quat(matrices: ArrayLike, return_torch: bool = False, device: torch.device = None):
    """
    Convert rotation matrices to quaternions.
    
    Args:
        matrices: (N, 3, 3) or (3, 3) rotation matrices.

    Returns:
        quats: (N, 4) or (4,) quaternions [x, y, z, w].
    """
    out = R.from_matrix(_to_numpy(matrices)).as_quat()
    return _to_torch(out, device) if return_torch else out