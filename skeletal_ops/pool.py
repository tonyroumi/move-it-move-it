import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class SkeletalPooling(nn.Module):
    """
    Skeletal pooling layer. 

    Pooling is applied to pairs of edges that connect to (joint) nodes of degree 2. 
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
        self.pooled_regions, self.new_edge_list, self.new_adj_list = self._compute_pooling(edge_list, last_pool)

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

    def _compute_pooling(self, edge_list, last_pool=False):
        from collections import defaultdict, deque

        # Build degree and adjacency
        degree = defaultdict(int)
        adj = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
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
                new_edges = edge_list
                continue

            # Handle odd-length path
            if len(seq) % 2 == 1:
                pooling_list.append([seq[0]])
                new_edges.append(edge_list[seq[0]])
                seq = seq[1:]

            # Pairwise pooling
            for i in range(0, len(seq), 2):
                pooling_list.append([seq[i], seq[i + 1]])
                new_edges.append([edge_list[seq[i]][0], edge_list[seq[i + 1]][1]])

        # Add global position pooling
        pooling_list.append([self.E - 1])

        new_adj_list = self._reconstruct_adj(new_edges)

        return pooling_list, new_edges, new_adj_list

    def _reconstruct_adj(self, edges, d=2):
        edge_mat = calc_edge_mat(edges)
        neighbor_list = []
        edge_num = len(edge_mat)
        for i in range(edge_num):
            neighbor = []
            for j in range(edge_num):
                if edge_mat[i][j] <= d:
                    neighbor.append(j)
            neighbor_list.append(neighbor)

        # add neighbor for global part
        global_part_neighbor = neighbor_list[0].copy()

        for i in global_part_neighbor:
            neighbor_list[i].append(edge_num)
        neighbor_list.append(global_part_neighbor)

        return neighbor_list
    
    def forward(self, x: torch.Tensor):
        output = self.weight @ x
        return output, self.pooled_regions, self.new_edge_list, self.new_adj_list

# I will later implement a pooler to assist with these operations. 
def calc_edge_mat(edges):
    edge_num = len(edges)
    # edge_mat[i][j] = distance between edge(i) and edge(j)
    edge_mat = [[100000] * edge_num for _ in range(edge_num)]
    for i in range(edge_num):
        edge_mat[i][i] = 0

    # initialize edge_mat with direct neighbor
    for i, a in enumerate(edges):
        for j, b in enumerate(edges):
            link = 0
            for x in range(2):
                for y in range(2):
                    if a[x] == b[y]:
                        link = 1
            if link:
                edge_mat[i][j] = 1

    # calculate all the pairs distance
    for k in range(edge_num):
        for i in range(edge_num):
            for j in range(edge_num):
                edge_mat[i][j] = min(edge_mat[i][j], edge_mat[i][k] + edge_mat[k][j])
    return edge_mat