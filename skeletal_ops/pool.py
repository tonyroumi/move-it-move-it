import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple

class SkeletalPooling(nn.Module):
    """
    Skeletal pooling layer. 

    Args
    ----
    adj_list  :   Dict[int, List[int]]
                  Original adjacency map.

    p         :   int
                  Maximum chain length per pooling region.

    mode      :   str       
                  Pooling mode: 'mean' or 'max'.
    """
    def __init__(
        self, 
        adj_list: Dict[int, List[int]],
        p: int = 2,
        mode: str = "mean",
        downsampling_params: Dict[str, Tuple[int]] = None
    ):
        super().__init__()
        self.adj = adj_list
        self.downsampling_params = downsampling_params
        self.p = p
        self.mode = mode

        # Precompute pooling regions and new adjacency
        self.pool_regions, self.new_adj = self._compute_pooling(self.adj, p)

    @staticmethod
    def _find_chains_and_split(adj: Dict[int, List[int]], p: int) -> List[List[int]]:
        def degree(n): 
            return len(adj.get(n, []))
        
        visited = set()
        chains = []

        for node in sorted(adj.keys()):
            # Only start from nodes that are degree 2 and not yet visited
            if node in visited or degree(node) != 2:
                continue

            chain = [node]
            visited.add(node)

            # Walk backward until reaching a node whose degree != 2
            prev = node
            curr = adj[node][0]
            while curr not in visited and degree(curr) == 2:
                chain.insert(0, curr)
                visited.add(curr)
                nxt = [n for n in adj[curr] if n != prev][0]
                prev, curr = curr, nxt

            # Walk forward similarly
            prev = node
            curr = adj[node][1]
            while curr not in visited and degree(curr) == 2:
                chain.append(curr)
                visited.add(curr)
                nxt = [n for n in adj[curr] if n != prev][0]
                prev, curr = curr, nxt

            # Now ensure the chain is bounded by non-degree-2 joints
            # Include those boundary nodes to preserve connectivity
            head, tail = chain[0], chain[-1]
            if degree(head) != 1:  # possible root or branching point
                parents = [n for n in adj[head] if n not in chain]
                if parents:
                    chain.insert(0, parents[0])
            if degree(tail) != 1:
                children = [n for n in adj[tail] if n not in chain]
                if children:
                    chain.append(children[0])

            chains.append(chain)

        # Split long chains into ≤ p chunks
        pool_regions = []
        for chain in chains:
            N = len(chain)
            if N <= p:
                pool_regions.append(chain)
                continue

            full = N // p
            rem = N % p

            if rem == 0:
                for i in range(full):
                    pool_regions.append(chain[i * p:(i + 1) * p])
            else:
                # remainder belongs to the region closest to the root (front)
                # -> remainder at the start of the chain
                pool_regions.append(chain[:rem])
                start = rem
                for i in range(full):
                    pool_regions.append(chain[start:start + p])
                    start += p
        return pool_regions

    @staticmethod
    def _collapse_adj(adj: Dict[int, List[int]], pool_regions: List[List[int]]) -> Dict[int, List[int]]:
        mapping = {}
        for pid, region in enumerate(pool_regions):
            for n in region:
                mapping[n] = pid
        
        new_adj = {}
        for i, nbrs in adj.items():
            i_new = mapping.get(i, i)
            for j in nbrs:
                j_new = mapping.get(j, j)
                if i_new == j_new:
                    continue
                new_adj.setdefault(i_new, set()).add(j_new)
        
        return {i: sorted(list(v)) for i, v in new_adj.items()}

    @classmethod
    def _compute_pooling(cls, adj: Dict[int, List[int]], p: int):
        pool_regions = cls._find_chains_and_split(adj, p)
        new_adj = cls._collapse_adj(adj, pool_regions)
        return pool_regions, new_adj
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[List[int]], Dict[int, List[int]]]:
        #Static branch pooling
        if x.dim() == 3:
            pooled = []
            for region in self.pool_regions:
                subset = x[:, region, :]
                if self.mode == "mean":
                    pooled.append(subset.mean(dim=1))
                else:
                    pooled.append(subset.max(dim=1).values)
            pooled = torch.stack(pooled, dim=1)                     #[B, R, C]
        
        #Dynamic branch
        else:
            pooled = []
            for region in self.pool_regions:
                subset = x[:, :, region, :]                         # [B,T,len(region),C]
                if self.mode == "mean":
                    pooled.append(subset.mean(dim=2))
                else:
                    pooled.append(subset.max(dim=2).values)
            pooled = torch.stack(pooled, dim=2)                     # [B,T,R,C]

            #Temporal downsampling
            pooled = F.avg_pool2d(pooled.permute(0, 3, 2, 1), 
                                  **(self.downsampling_params)).permute(0, 3, 2, 1)
        
        return pooled, self.pool_regions, self.new_adj

