from .array import ArrayLike, ArrayUtils

from src.skeletal_models import SkeletonTopology
from typing import List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import math

class SkeletonUtils:
    """ Skeleton utiities """

    @staticmethod
    def prune_joints(data: ArrayLike, cutoff: int, discard_root: int = False) -> ArrayLike:
        """ Prune joints based on an index cutoff. """
        if ArrayUtils.is_tensor(data):
            idx = torch.arange(cutoff, device=data.device)
        else:
            idx = np.arange(cutoff)

        idx = idx[discard_root:]
        # Joint-major: (J, ...)
        if data.ndim == 2:
            return data[idx]

        # Batch-major: (N, J, ...)
        elif data.ndim == 3:
            return data[:, idx]

    @staticmethod
    def construct_adj(edge_list: ArrayLike, d: int = 2, global_edges: bool = True):
        """
        Returns adjacency list of edges within distance <= d
        """
        def add_global_edge(neighbors):
            E = len(neighbors)-1
            global_idx = E

            neighbors.append(list(range(E)))  # global connects to all
            for i in range(E):
                neighbors[i].append(global_idx)

            return neighbors

        dist = SkeletonUtils.calc_edge_distance_matrix(edge_list)
        E = len(edge_list)

        neighbors = []
        for i in range(E):
            nbrs = [j for j in range(E) if dist[i][j] <= d]
            neighbors.append(nbrs)
        
        if global_edges:
            neighbors = add_global_edge(neighbors)

        return neighbors
    
    @staticmethod
    def calc_edge_distance_matrix(edges):
        """
        edges: List[Tuple[int, int]]
        returns: (E, E) matrix of shortest edge–edge distances
        """
        E = len(edges)
        INF = math.inf

        dist = [[INF] * E for _ in range(E)]
        for i in range(E):
            dist[i][i] = 0

        # distance 1 if edges share a joint
        for i, (a0, a1) in enumerate(edges):
            for j, (b0, b1) in enumerate(edges):
                if i == j:
                    continue
                if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                    dist[i][j] = 1

        # Floyd–Warshall (E is small for skeletons)
        for k in range(E):
            for i in range(E):
                for j in range(E):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    @staticmethod
    def get_ee_velocity(
        positions: torch.Tensor,
        topologies: Tuple[SkeletonTopology,SkeletonTopology]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Compute the velocity of end effectors """
        ee_vels = []
        for i in range(len(topologies)):
            vel = positions[i][:, 1:] - positions[i][:, :-1]  
            ee_vel = vel[:, :, topologies[i].ee_ids, :]  
            ee_vels.append(ee_vel) 
        
        return ee_vels              
