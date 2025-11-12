import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class SkeletalPooling(nn.Module):
    """
    Skeletal pooling layer. 

    Pooling is applied to pairs of edges that connect to (joint) nodes of degree 2. 

    Args
    ----
    edge_list: A list of edges described by (parent, child) joint nodes. 

    channels_per_edge: Channels per edge lol.

    last_pool: Whether or not to pool edges.
    """
    def __init__(
        self, 
        edge_list: List[Tuple[int, int]],
        channels_per_edge: int,
        last_pool: bool = False
    ):
        super().__init__()
        self.edge_list = edge_list
        self.E = len(self.edge_list) + 1
        self.channels_per_edge = channels_per_edge

        # Precompute pooling regions and new adjacency
        self.pooled_regions, self.new_edge_list = self._compute_pooling(edge_list, last_pool)

        self._init_net()
    
    def _init_net(self):
        rows = len(self.pooled_regions) * self.channels_per_edge
        cols = self.E * self.channels_per_edge

        # Pooling operation
        # Block-diagonal averaging matrix that groups edges into pooled regions:
        # weight: (region-to-pool-features, edge-features)
        weight = torch.zeros(rows, cols)
        for i, group in enumerate(self.pooled_regions):
            scale = 1.0 / len(group)
            for j in group:
                idx = torch.arange(self.channels_per_edge)
                weight[i * self.channels_per_edge + idx, j * self.channels_per_edge + idx] = scale

        self.weight = nn.Parameter(weight, requires_grad=False)

    def _compute_pooling(self, edges, last_pool=False):
        from collections import defaultdict, deque

        # Build degree and adjacency
        degree = defaultdict(int)
        adj = defaultdict(list)
        for idx, (u, v) in enumerate(edges):
            degree[u] += 1
            degree[v] += 1
            adj[u].append((v, idx))

        seq_list = []
        visited = set()

        # DFS to collect edge sequences
        def dfs(node, seq):
            if node in visited:
                return
            visited.add(node)

            # Stop sequence at branching or leaf
            if degree[node] > 2 and node != 0:
                seq_list.append(seq)
                seq = []
            elif degree[node] == 1 and seq:
                seq_list.append(seq)
                return

            for nxt, e_idx in adj[node]:
                if e_idx not in {i for s in seq_list for i in s}:  # prevent reuse
                    dfs(nxt, seq + [e_idx])

        dfs(0, [])

        pooling_list, new_edges = [], []
        for seq in seq_list:
            if last_pool:
                pooling_list.append(seq)
                continue

            # Handle odd-length path
            if len(seq) % 2 == 1:
                pooling_list.append([seq[0]])
                new_edges.append(edges[seq[0]])
                seq = seq[1:]

            # Pairwise pooling
            for i in range(0, len(seq), 2):
                pooling_list.append([seq[i], seq[i + 1]])
                new_edges.append([edges[seq[i]][0], edges[seq[i + 1]][1]])

        # Add global position pooling
        pooling_list.append([self.E - 1])

        return pooling_list, new_edges
    
    def forward(self, x: torch.Tensor):
        return self.weight @ x
