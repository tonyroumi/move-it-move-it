from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class SkeletonTopology:
    edge_topology: torch.Tensor = None
    edge_adjacency: torch.Tensor = None
    ee_ids: torch.Tensor = None

@dataclass
class MotionOutput:
    latents: torch.Tensor
    motion: torch.Tensor          # normalized rotations
    rotations: torch.Tensor       # denormalized rotations
    positions: torch.Tensor       # FK world positions
    ee_vels: torch.Tensor = None  # end-effector velocities