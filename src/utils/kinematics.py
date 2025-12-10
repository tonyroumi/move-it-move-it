from pickletools import int4
import torch

from typing import List, Tuple

class ForwardKinematics:
   def __init__(self, topology: List[Tuple[int]]):
      self.topology = torch.tensor(topology, dtype=torch.int64).tolist()

   def forward(
      self,
      quaternions: torch.Tensor,
      offsets: torch.Tensor,
      root_pos: torch.Tensor,
      world: bool = False
   ):
    """
    Forward kinematics for both batched and unbatched quaternion data.
    Quaternions and offsets must share the same leading batch dims.

    Args:
      quaternions: (..., 4) quaternions in (w, x, y, z) format.
      offsets: (..., 3) joint offsets
      root_pos: (..., 3) global root translation
      world: Compute positions of joints in the world frame.

    Returns:
      P: (..., J, 3) joint positions
    """
    added_batch = False
    if quaternions.dim() == 2:
        quaternions = quaternions.unsqueeze(0)
        offsets = offsets.unsqueeze(0)
        root_pos = root_pos.unsqueeze(0)
        added_batch = True

    *batch_dims, J, _ = quaternions.shape
    
    P = torch.zeros(*batch_dims, J, 3, device=quaternions.device)
    R = torch.zeros(*batch_dims, J, 3, 3, device=quaternions.device)

    P[..., 0, :] = root_pos

    rotmats = self.quat_to_rotmat(quaternions)

    for (parent, joint) in self.topology:

        R[..., joint,: ,:] = R[..., parent, :, :] @ rotmats[..., joint, :, :]

        local_pos = (R[..., parent, :, :] @ offsets[:, joint, None, :, None]).squeeze(-1)

        # if world:
        #     P[..., joint] = P[..., parent] + local_pos.unsqueeze(-2)
        # else:
        P[..., joint, :] = local_pos

    if added_batch:
        P = P.squeeze(0)

    return P

   def quat_to_rotmat(self, quaternions: torch.Tensor, pad_root: bool = True) -> torch.Tensor:
    """     
    Convert quaternions to rotation matrices.

    Args:
        quaternions: (..., 4) quaternions in (w, x, y, z) format.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    w, x, y, z = quaternions.unbind(-1)  

    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z

    R = torch.empty(quaternions.shape[:-1] + (3, 3), device=quaternions.device)

    R[..., 0, 0] = 1 - 2*(yy + zz)
    R[..., 0, 1] =     2*(xy - wz)
    R[..., 0, 2] =     2*(xz + wy)

    R[..., 1, 0] =     2*(xy + wz)
    R[..., 1, 1] = 1 - 2*(xx + zz)
    R[..., 1, 2] =     2*(yz - wx)

    R[..., 2, 0] =     2*(xz - wy)
    R[..., 2, 1] =     2*(yz + wx)
    R[..., 2, 2] = 1 - 2*(xx + yy)

    return R
