from typing import Dict, List
import torch
from skeletal_ops import SkeletalPooling


# Simple linear skeleton with degree-2 nodes
ADJ = {
    0: [1],
    1: [0, 2],
    2: [1, 3],
    3: [2, 4],
    4: [3],
}

def test_no_remainders():
    """Case 1: Chain divides evenly into pooling groups."""
    # Linear chain: 0-1-2-3-4-5, each degree-2 except ends
    adj = {0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3,5], 5:[4]}
    pool = SkeletalPooling(adj, p=2, downsampling_params={"kernel_size":(1, 2), "stride":(1, 2)}, mode="mean")
    
    pool_regions, new_adj = pool.pool_regions, pool.new_adj
    # Expect 3 pooled regions of length 2
    assert all(len(r) == 2 for r in pool_regions)
    assert len(pool_regions) == 3
    # Adjacency between regions should be sequential
    keys_sorted = sorted(new_adj.keys())
    assert all(k+1 in new_adj[k] or k-1 in new_adj[k] for k in keys_sorted if new_adj[k])

    visualize_pooling(adj, new_adj, "test_no_remainders.png")


def test_with_some_remainders():
    """Case 2: Chain length not divisible by p -> remainder at root."""
    adj = {0:[1], 1:[0,2], 2:[1,3], 3:[2,4], 4:[3]}
    pool = SkeletalPooling(adj, p=2, downsampling_params={"kernel_size":(1, 2), "stride":(1, 2)}, mode="mean")
    
    pool_regions, new_adj = pool.pool_regions, pool.new_adj
    # Expect one region with remainder length 1 (closest to root)
    lengths = [len(r) for r in pool_regions]
    assert 1 in lengths, "Should contain remainder region near root"
    # Adjacency between pooled regions still valid
    for k,v in new_adj.items():
        assert isinstance(v, list)
    
    visualize_pooling(adj, new_adj, "test_with_some_remainders.png")


def test_with_large_remainders():
    """Case 3: Long chain with many segments and leftover remainder."""
    N = 11
    adj = {i:[i-1,i+1] for i in range(1,N-1)}
    adj[0] = [1]
    adj[N-1] = [N-2]

    pool = SkeletalPooling(adj, p=3, downsampling_params={"kernel_size":(1, 2), "stride":(1, 2)}, mode="mean")
    pool_regions, new_adj = pool.pool_regions, pool.new_adj

    # Remainder should be < p and placed near root (smallest index)
    lengths = [len(r) for r in pool_regions]
    assert any(l < 3 for l in lengths)
    remainder_region = pool_regions[0]
    assert min(remainder_region) == 0, "Remainder should include root end of chain"
    assert all(isinstance(k, int) for k in new_adj)

    visualize_pooling(adj, new_adj, "test_with_large_remainders.png")


def test_static_mean_pooling():
    """Verify mean pooling on a static input and correct new adjacency structure."""
    pool = SkeletalPooling(ADJ, p=2, downsampling_params={"kernel_size":(1, 2), "stride":(1, 2)}, mode="mean")

    x = torch.tensor(
        [[[1., 1., 1.],   # node 0
          [2., 2., 2.],
          [3., 3., 3.],
          [4., 4., 4.],
          [5., 5., 5.]]]
    )  # [B=1, J=5, C=3]

    pooled, regions, new_adj = pool(x)

    # Check shape
    assert pooled.ndim == 3
    assert pooled.shape[0] == 1  # batch
    assert pooled.shape[2] == 3  # channels

    # Check region structure
    assert all(isinstance(r, list) for r in regions)
    assert isinstance(new_adj, dict)

    # Verify mean pooling correctness
    for r_idx, region in enumerate(regions):
        expected = x[:, region, :].mean(dim=1)
        torch.testing.assert_close(pooled[:, r_idx, :], expected)

    # Ensure adjacency collapsed
    assert all(isinstance(v, list) for v in new_adj.values())


def test_static_max_pooling():
    """Verify max pooling gives maximum values per pooling region."""
    pool = SkeletalPooling(ADJ, p=2, downsampling_params={"kernel_size":(1, 2), "stride":(1, 2)}, mode="max")

    x = torch.arange(5 * 3).float().reshape(1, 5, 3)  # increasing values per node

    pooled, regions, _ = pool(x)

    for r_idx, region in enumerate(regions):
        expected = x[:, region, :].max(dim=1).values
        torch.testing.assert_close(pooled[:, r_idx, :], expected)


def test_dynamic_mean_pooling():
    """Verify dynamic mean pooling aggregates across time and preserves temporal dimension."""
    pool = SkeletalPooling(ADJ, p=2, downsampling_params={"kernel_size":(1, 2), "stride":(1, 2)}, mode="mean")

    B, T, J, C = 2, 4, 5, 3
    x = torch.randn(B, T, J, C)

    pooled, regions, new_adj = pool(x)

    # Shape checks
    assert pooled.shape == (B, T // 2, len(regions), C)
    assert isinstance(new_adj, dict)
    assert len(new_adj) <= len(ADJ)

    # Check that pooling reduces along joint dimension only
    assert torch.allclose(
        pooled.mean(dim=(1, 2, 3)), x.mean(dim=(1, 2, 3)), atol=1e-1
    ), "Global mean of values should roughly remain in same scale after pooling"

def visualize_pooling(before: Dict[int, List[int]],
                      after: Dict[int, List[int]],
                      save_path: str = "graph_pooling.png"):
    import networkx as nx
    import matplotlib.pyplot as plt
    """
    Visualize graph before and after pooling.

    Args
    ----
    before    : Dict[int, List[int]]
                Original adjacency list.

    after     : Dict[int, List[int]]
                Pooled adjacency list.

    save_path : str
                File path to save the resulting image.
    """
    def build_graph(adj):
        G = nx.Graph()
        for node, nbrs in adj.items():
            for n in nbrs:
                G.add_edge(node, n)
        return G

    G_before = build_graph(before)
    G_after = build_graph(after)

    # consistent layout for better visual comparison
    pos_before = nx.spring_layout(G_before, seed=42)
    pos_after = nx.spring_layout(G_after, seed=42)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    nx.draw(G_before, pos_before,
            ax=axes[0],
            with_labels=True,
            node_color="lightblue",
            node_size=600,
            edge_color="gray",
            font_weight="bold")
    axes[0].set_title("Before Pooling")
    axes[0].axis("off")

    nx.draw(G_after, pos_after,
            ax=axes[1],
            with_labels=True,
            node_color="lightgreen",
            node_size=600,
            edge_color="gray",
            font_weight="bold")
    axes[1].set_title("After Pooling")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison figure to {save_path}")

def run_tests():
    test_no_remainders()
    test_with_some_remainders()
    test_with_large_remainders()
    test_static_mean_pooling()
    test_static_max_pooling()
    test_dynamic_mean_pooling()

if __name__ == "__main__":
    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")

    run_tests()
    print("All tests passed!")