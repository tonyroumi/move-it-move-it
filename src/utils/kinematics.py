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
        Forward kinematics for unbatched quaternion data.

        Args:
            quaternions: (T, num_joints, 4) quaternions in (w, x, y, z) format
            offsets: (num_joints, 3) joint offsets
            root_pos: (T, 3) global root translation
            world: Compute positions of joints in the world frame

        Returns:
            P: (T, num_joints, 3) joint positions
        """
        T, J, _ = quaternions.shape
        
        P = torch.zeros(T, J, 3, device=quaternions.device)
        R = torch.zeros(T, J, 3, 3, device=quaternions.device)

        rotmats = self.quat_to_rotmat(quaternions)

        P[:, 0, :] = root_pos
        R[:, 0, :, :] = rotmats[:, 0, :, :]

        for (parent, joint) in self.topology:
            R[:, joint, :, :] = R[:, parent, :, :] @ rotmats[:, joint, :, :]
            
            local_pos = (
                R[:, parent, :, :]
                @ offsets[joint, :, None]
            ).squeeze(-1)

            P[:, joint, :] = P[:, parent, :] + local_pos

        return P
        
   def forward_batched(
        self,
        quaternions: torch.Tensor,
        offsets: torch.Tensor,
        root_pos: torch.Tensor,
        world: bool = False
   ):
       """
       Forward kinematics for batched quaternion data.
       Args:
           quaternions: (B, T, num_joints, 4) quaternions in (w, x, y, z) format
           offsets: (B, num_joints, 3) joint offsets
           root_pos: (B, T, 3) global root translation
           world: Compute positions of joints in the world frame
       Returns:
           P: (B, T, num_joints, 3) joint positions
       """
       B, T, J, _ = quaternions.shape
       
       P = torch.zeros(B, T, J, 3, device=quaternions.device)
       R = torch.zeros(B, T, J, 3, 3, device=quaternions.device)

       rotmats = self.quat_to_rotmat(quaternions)

       P[:, :, 0, :] = root_pos
       R[:, :, 0, :, :] = rotmats[:, :, 0, :, :]

       for (parent, joint) in self.topology:
           R[:, :, joint, :, :] = R[:, :, parent, :, :] @ rotmats[:, :, joint, :, :]
           
           local_pos = (
               R[:, :, parent, :, :]
               @ offsets[:, joint, :, None]
           ).squeeze(-1)
           P[:, :, joint, :] = P[:, :, parent, :] + local_pos
       return P

   def quat_to_rotmat(self, quaternions: torch.Tensor, pad_root: bool = True) -> torch.Tensor:
    """     
    Convert quaternions to rotation matrices.

    Args:
        quaternions: (..., 4) quaternions in (x, y, z, w) format.

    Returns:
        (..., 3, 3) rotation matrices.
    """
    quaternions = quaternions / quaternions.norm(dim=-1, keepdim=True)

    x, y, z, w = quaternions.unbind(-1)  

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
