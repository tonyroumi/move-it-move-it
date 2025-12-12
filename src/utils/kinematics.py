import torch
from typing import List, Tuple

class ForwardKinematics:
    """Forward kinematics utilities."""

    @staticmethod
    def forward(
        quaternions: torch.Tensor,
        offsets: torch.Tensor,
        root_pos: torch.Tensor,
        topology: List[Tuple[int]],
    ) -> torch.Tensor:
        """Compute joint positions for unbatched motion."""
        T, J, _ = quaternions.shape

        P = torch.zeros(T, J, 3, device=quaternions.device)
        R = torch.zeros(T, J, 3, 3, device=quaternions.device)

        rotmats = ForwardKinematics.quat_to_rotmat(quaternions)

        P[:, 0, :] = root_pos 
        R[:, 0, :, :] = rotmats[:, 0, :, :] 
        
        for (parent, joint) in topology:
            R[:, joint, :, :] = R[:, parent, :, :] @ rotmats[:, joint, :, :] 
            local_pos = ( R[:, parent, :, :] @ offsets[joint-1, :, None] ).squeeze(-1) 
            
            P[:, joint, :] = P[:, parent, :] + local_pos 
        
        return P

    @staticmethod
    def forward_batched(
        quaternions: torch.Tensor,
        offsets: torch.Tensor,
        root_pos: torch.Tensor,
        topology: List[Tuple[int]],
    ) -> torch.Tensor:
        """Compute joint positions for batched motion."""
        B, T, J, _ = quaternions.shape

        P = torch.zeros(B, T, J, 3, device=quaternions.device) 
        R = torch.zeros(B, T, J, 3, 3, device=quaternions.device) 
        
        rotmats = ForwardKinematics.quat_to_rotmat(quaternions) 
        
        P[:, :, 0, :] = root_pos 
        R[:, :, 0, :, :] = rotmats[:, :, 0, :, :] 
        
        for (parent, joint) in topology: 
            R[:, :, joint, :, :] = R[:, :, parent, :, :] @ rotmats[:, :, joint, :, :] 
            local_pos = ( R[:, :, parent, :, :] @ offsets[joint-1, :, None] ).squeeze(-1) 
            
            P[:, :, joint, :] = P[:, :, parent, :] + local_pos 
        return P

    @staticmethod
    def quat_to_rotmat(quaternions: torch.Tensor) -> torch.Tensor:
        """Convert unit quaternions to rotation matrices."""
        q = quaternions / quaternions.norm(dim=-1, keepdim=True)
        x, y, z, w = q.unbind(-1)

        R = torch.empty(q.shape[:-1] + (3, 3), device=q.device)

        R[..., 0, 0] = 1 - 2*(y*y + z*z)
        R[..., 0, 1] = 2*(x*y - w*z)
        R[..., 0, 2] = 2*(x*z + w*y)
        R[..., 1, 0] = 2*(x*y + w*z)
        R[..., 1, 1] = 1 - 2*(x*x + z*z)
        R[..., 1, 2] = 2*(y*z - w*x)
        R[..., 2, 0] = 2*(x*z - w*y)
        R[..., 2, 1] = 2*(y*z + w*x)
        R[..., 2, 2] = 1 - 2*(x*x + y*y)

        return R