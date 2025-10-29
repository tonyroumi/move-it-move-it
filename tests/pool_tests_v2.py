import torch
from skeletal_ops import SkeletalPooling


def make_tensor(B=1, N=6, C=4, dynamic=False, T=3):
    if dynamic:
        return torch.randn(B, N, T, C)
    return torch.randn(B, N, C)


def test_simple_chain_no_split():
    adj = {0: [1], 1: [0, 2], 2: [1]}
    pool = SkeletalPooling(adj, p=3)
    assert len(pool.pool_regions) == 1
    print("✓ simple_chain_no_split passed")


def test_chain_split_even():
    adj = {i: [i - 1, i + 1] for i in range(1, 5)}
    adj[0] = [1]; adj[5] = [4]
    pool = SkeletalPooling(adj, p=2)
    expected = [[0, 1], [2, 3], [4, 5]]
    assert pool.pool_regions == expected
    print("✓ chain_split_even passed")


def test_chain_with_remainder_root_side():
    adj = {i: [i - 1, i + 1] for i in range(1, 4)}
    adj[0] = [1]; adj[4] = [3]
    pool = SkeletalPooling(adj, p=2)
    expected = [[0], [1, 2], [3, 4]]
    assert pool.pool_regions == expected
    print("✓ chain_with_remainder_root_side passed")


def test_branch_structure_two_chains():
    adj = {0: [1, 2], 1: [0, 3], 2: [0, 4], 3: [1], 4: [2]}
    pool = SkeletalPooling(adj, p=2)
    regions = pool.pool_regions

    # 0 must appear in exactly one region
    containing = [r for r in regions if 0 in r]
    assert len(containing) == 1
    r0 = containing[0]

    # 0 grouped with exactly one neighbor (either 1 or 2), not both
    assert (1 in r0) ^ (2 in r0)
    assert not ((1 in r0) and (2 in r0))
    print("✓ branch_structure_two_chains passed")


def test_internal_branch_nodes_are_not_chained():
    adj = {0: [1], 1: [0, 2, 3], 2: [1, 4], 3: [1], 4: [2]}
    pool = SkeletalPooling(adj, p=2)
    for chain in pool.pool_regions:
        assert not (0 in chain and 2 in chain)
    print("✓ internal_branch_nodes_are_not_chained passed")


def test_collapse_adj_output_structure():
    adj = {0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}
    pool_regions = [[0, 1], [2, 3]]
    new_adj = SkeletalPooling._collapse_adj(adj, pool_regions)
    assert new_adj == {0: [1], 1: [0]}
    print("✓ collapse_adj_output_structure passed")


def test_forward_static_mean():
    adj = {0: [1], 1: [0, 2], 2: [1]}
    pool = SkeletalPooling(adj, p=2, mode="mean")
    x = make_tensor(B=2, N=3, C=4)
    pooled, regions, _ = pool(x)
    assert pooled.shape[1] == len(regions)
    print("✓ forward_static_mean passed")


def test_forward_static_max():
    adj = {0: [1], 1: [0, 2], 2: [1]}
    pool = SkeletalPooling(adj, p=2, mode="max")
    x = make_tensor(B=2, N=3, C=4)
    pooled, regions, _ = pool(x)
    assert pooled.shape[1] == len(regions)
    print("✓ forward_static_max passed")


def test_forward_dynamic_branch():
    adj = {0: [1], 1: [0, 2], 2: [1]}
    pool = SkeletalPooling(adj, p=2, mode="mean", downsampling_params={"kernel_size": (1, 1)})
    x = make_tensor(B=1, N=3, C=4, dynamic=True, T=3)
    pooled, regions, _ = pool(x)
    assert pooled.dim() == 4
    assert pooled.shape[2] == len(regions)
    print("✓ forward_dynamic_branch passed")


def test_edge_case_isolated_nodes():
    adj = {0: [], 1: [], 2: [3], 3: [2]}
    pool = SkeletalPooling(adj, p=2)
    for region in pool.pool_regions:
        assert len(region) >= 1
    print("✓ edge_case_isolated_nodes passed")


def test_remainder_exact_multiple():
    adj = {i: [i - 1, i + 1] for i in range(1, 3)}
    adj[0] = [1]; adj[3] = [2]
    pool = SkeletalPooling(adj, p=2)
    assert all(len(r) == 2 for r in pool.pool_regions)
    print("✓ remainder_exact_multiple passed")


def test_long_chain_stress():
    adj = {i: [i - 1, i + 1] for i in range(1, 19)}
    adj[0] = [1]; adj[19] = [18]
    pool = SkeletalPooling(adj, p=4)
    expected_total = (20 // 4) + (1 if 20 % 4 else 0)
    assert len(pool.pool_regions) == expected_total
    print("✓ long_chain_stress passed")


def test_output_consistency_between_modes():
    adj = {0: [1], 1: [0, 2], 2: [1]}
    pool_mean = SkeletalPooling(adj, p=2, mode="mean")
    pool_max = SkeletalPooling(adj, p=2, mode="max")
    x = make_tensor(B=2, N=3, C=4)
    y1, _, _ = pool_mean(x)
    y2, _, _ = pool_max(x)
    assert y1.shape == y2.shape
    print("✓ output_consistency_between_modes passed")


if __name__ == "__main__":
    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")

    tests = [
        test_simple_chain_no_split,
        test_chain_split_even,
        test_chain_with_remainder_root_side,
        test_branch_structure_two_chains,
        test_internal_branch_nodes_are_not_chained,
        test_collapse_adj_output_structure,
        test_forward_static_mean,
        test_forward_static_max,
        test_forward_dynamic_branch,
        test_edge_case_isolated_nodes,
        test_remainder_exact_multiple,
        test_long_chain_stress,
        test_output_consistency_between_modes,
    ]
    for t in tests:
        t()
    print("\nAll tests passed successfully.")
