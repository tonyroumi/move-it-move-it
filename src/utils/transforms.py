from .rotation import axis_angle_to_matrix, matrix_to_quat
from .data import ArrayLike, _is_tensor, _to_torch, _to_numpy

from typing import Union, List
import numpy as np

def global_aa_to_local_quat(
    axis_angles: ArrayLike,
    parent_kintree: Union[np.ndarray, List[int]],
    include_root: bool = False,
) -> ArrayLike:
    """
    Conversion of global axis-angle rotations to local quaternions.
    
    Args:
        axis_angles: (J, 3) global axis-angle rotations.
        parent_kintree: (J,) parent indices
        include_root: whether to include the untransformed root in the result. 
                      
    Returns:
        local_quats: (J, 4) local joint rotations as quaternions [x, y, z, w].

    """ 
    is_tensor = _is_tensor(axis_angles)
    original_input = axis_angles
    
    # Convert to numpy for scipy operations
    aa_np = _to_numpy(axis_angles)
    parents_np =_to_numpy(parent_kintree)

    # Convert axis-angles to rotation matrices
    Rmats = axis_angle_to_matrix(aa_np)
    
    start_idx = 0 if include_root else 1
    num_joints = len(axis_angles)
    
    local_quats = []
    
    for i in range(start_idx, num_joints):
        parent_idx = parents_np[i]
        
        # Get parent and child rotation matrices
        R_parent = Rmats[parent_idx]
        R_child = Rmats[i]
        
        # Compute relative rotation of child
        R_rel = R_parent.T @ R_child
        
        # Convert to quaternion
        q_rel = matrix_to_quat(R_rel)
        local_quats.append(q_rel)
    
    result = _to_numpy(local_quats)
    
    if is_tensor:
        result = _to_torch(result, original_input.device)
    
    return result

def global_aa_to_local_quat_batched(
    axis_angles: ArrayLike,
    parent_kintree: Union[np.ndarray, list],
    include_root: bool = False
) -> ArrayLike:
    """
    Batched conversion of global axis-angle rotations to local quaternions.
    
    Args:
        axis_angles: (T, J, 3) global axis-angle rotations.
        parent_kintree: (J,) parent indices
        include_root: whether to include the untransformed root in the result. 
                      
    Returns:
        local_quats: (T, J, 4) local joint rotations as quaternions [x, y, z, w].
    """
    is_tensor = _is_tensor(axis_angles)
    orig_input = axis_angles

    aa = _to_numpy(axis_angles)          # (T, J, 3)
    parent_kintree = _to_numpy(parent_kintree)         # (J,)
    T, J, _ = aa.shape

    # Convert all global AA → rotation matrices in one call
    # result: (T*J, 3, 3)
    R_global = axis_angle_to_matrix(aa.reshape(-1, 3))
    R_global = R_global.reshape(T, J, 3, 3)

    local_quats = []

    start = 0 if include_root else 1
    local_quats = np.zeros((T, J-start, 4), dtype=np.float32)

    for j in range(start, J):
        p = parent_kintree[j]

        # R_rel = R_parent^T * R_child
        # both broadcast as (T, 3, 3)
        R_parent = R_global[:, p]          # (T, 3, 3)
        R_child  = R_global[:, j]          # (T, 3, 3)
        R_rel = np.einsum("tij,tjk->tik", R_parent.transpose(0,2,1), R_child) # Batched MM

        # Convert batch of R_rel to quats
        q_rel = matrix_to_quat(R_rel)      # (T, 4)

        local_quats[:,j-start] = q_rel

    result = local_quats
    if is_tensor:
        result = _to_torch(result, orig_input.device)

    return result
