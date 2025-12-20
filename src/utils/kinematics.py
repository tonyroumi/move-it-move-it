from turtle import forward
import torch
from typing import List, Tuple

from src.skeletal_models import SkeletonTopology

class ForwardKinematics:
    """Forward kinematics utilities."""

    @staticmethod
    def forward(
        quaternions: torch.Tensor,
        offsets: torch.Tensor,
        root_pos: torch.Tensor,
        topologies: List[SkeletonTopology],
        world: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute joint positions for unbatched motion.
        
        NOTE: This assume that the root quat is included.
        """
        #TODO: Handle no root quat.
        forward_positions = []
        for i in range(len(topologies)):
            T, J, _ = quaternions[i].shape

            P = torch.zeros(T, J, 3, device=quaternions[i].device)
            R = torch.zeros(T, J, 3, 3, device=quaternions[i].device)

            rotmats = ForwardKinematics.quat_to_rotmat(quaternions[i])

            P[:, 0, :] = root_pos[i]
            R[:, 0, :, :] = rotmats[:, 0, :, :] 
            
            for (parent, child) in topologies[i].edge_topology:
                R[:, child, :, :] = R[:, parent, :, :] @ rotmats[:, child, :, :] 
                P[:, child, :] = ( R[:, parent, :, :] @ offsets[i][child-1, :, None] ).squeeze(-1) 

                if world:
                    P[:, child, :] += P[:, parent, :]
                forward_positions.append(P)
        
        return forward_positions

    @staticmethod
    def forward_batched(
        quaternions: torch.Tensor,
        offsets: torch.Tensor,
        root_pos: torch.Tensor,
        topologies: Tuple[SkeletonTopology, SkeletonTopology],
        world: bool = True,
    ) -> torch.Tensor:
        """
        Compute joint positions for batched motion.

        NOTE: This assume that the root quat is included.
        """
        #TODO: Handle no root quat.
        forward_positions = []
        for i in range(len(topologies)):
            B, T, J, _ = quaternions[i].shape # Root orientation not included here

            P = torch.zeros(B, T, J+1, 3, device=quaternions[i].device) 
            R = torch.zeros(B, T, J+1, 3, 3, device=quaternions[i].device) 
            
            rotmats = ForwardKinematics.quat_to_rotmat(quaternions[i]) 
            
            P[:, :, 0, :] = root_pos[i].transpose(1,2) 
            R[:, :, 0, :, :] = torch.eye(3, device=R.device, dtype=R.dtype)
            
            for (parent, child) in topologies[i].edge_topology: 
                R[:, :, child, :, :] = R[:, :, parent, :, :] @ rotmats[:, :, child-1, :, :] 
                P[:, :, child, :] = ( R[:, :, parent, :, :] @ offsets[i][:, None, child, :, None]).squeeze(-1) 

                if world:
                    P[:, :, child, :] += P[:, :, parent, :]
                
                forward_positions.append(P)

        return forward_positions
    
    @staticmethod
    def local_to_world(positions: torch.Tensor, topologies: Tuple[SkeletonTopology, SkeletonTopology]):
        """ Convert from local positions to world """
        world_positions = []
        for i in range(len(topologies)):
            positions = positions[0].clone()
            for parent, child in topologies[i].edge_topology:
                positions[:, child, :] += positions[:, parent, :]

            world_positions.append(positions)
        return positions

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