from typing import Tuple, Union
import math
import numpy as np
import torch

from src.utils import ArrayLike, ArrayUtils
from src.core.types import SkeletonTopology

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
    def construct_adj(
        edge_list: ArrayLike,
        d: int = 2,
        global_edges: bool = True,
        global_edge_source: int | None = None,
    ):
        """
        Returns adjacency list of edges within distance <= d
        """
        def add_global_edge(neighbors):
            E = len(neighbors)
            global_idx = E

            if global_edge_source is None:
                global_neighbors = list(range(E))
            else:
                global_neighbors = neighbors[global_edge_source].copy()

            neighbors.append(global_neighbors)
            for i in global_neighbors:
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
        positions: ArrayLike,
        topology: Union[SkeletonTopology] 
    ) -> Tuple[ArrayLike, ArrayLike]:
        """ Compute the velocity of end effectors """
        vel = positions[:, 1:] - positions[:, :-1]  
        ee_vel = vel[:, :, topology.ee_ids, :]  
        return ee_vel              
