import torch
from skeletal_ops import SkeletalPooling, SkeletalUnpooling


def make_static_input():
    # 3 nodes, 8 features per node
    x = torch.arange(1, 25, dtype=torch.float32).view(1, 3, 8)
    # shape: [B=1, N=3, C=8]
    return x


def make_dynamic_input():
    # 3 nodes, 2 time steps, 4 features
    x = torch.arange(1, 25, dtype=torch.float32).view(1, 2, 3, 4)
    # shape: [B=1, T=2, N=3, C=4]
    return x


def test_static_pool_unpool_identity_mean():
    """
    Pool + unpool with mode='mean' should reproduce node-wise means correctly.
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}
    x = make_static_input()

    # Pool step
    pool = SkeletalPooling(adj, p=2, mode="mean")
    pooled, regions, new_adj = pool(x)

    # Unpool step
    unpool = SkeletalUnpooling(adj, regions)
    x_unpooled = unpool(pooled)

    # ---- Expected behavior ----
    # Region [0,1]: mean(x0, x1)
    region01_mean = x[:, [0], :]
    # Region [2]: x2 unchanged
    region2 = x[:, [1,2], :].mean(dim=1, keepdim=True)

    expected_unpooled = torch.cat(
        [region01_mean, region2.repeat(1, 2, 1)], dim=1
    )

    assert torch.allclose(x_unpooled, expected_unpooled, atol=1e-6)
    print("✓ test_static_pool_unpool_identity_mean passed")


def test_static_pool_unpool_max():
    """
    Pool + unpool with mode='max' should reproduce node-wise max correctly.
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}
    x = make_static_input()

    pool = SkeletalPooling(adj, p=2, mode="max")
    pooled, regions, new_adj = pool(x)

    unpool = SkeletalUnpooling(adj, regions)
    x_unpooled = unpool(pooled)

    region01_max = x[:, [0], :]
    region2 = x[:, [1,2], :].max(dim=1, keepdim=True).values
    expected_unpooled = torch.cat(
        [region01_max, region2.repeat(1, 2, 1)], dim=1
    )

    assert torch.allclose(x_unpooled, expected_unpooled, atol=1e-6)
    print("✓ test_static_pool_unpool_max passed")


def test_dynamic_pool_unpool_mean_no_upsampling():
    """
    Dynamic branch: Pool + Unpool (no temporal upsampling).
    Should reconstruct region-based features correctly.
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}
    x = make_dynamic_input()

    pool = SkeletalPooling(adj, p=2, mode="mean", downsampling_params={"kernel_size": (1, 1)})
    pooled, regions, new_adj = pool(x)

    unpool = SkeletalUnpooling(adj, regions)
    x_unpooled = unpool(pooled)

    region01_mean = x[:, :, [0], :]
    region2 = x[:, :, [1,2], :].mean(dim=2, keepdim=True)
    expected_unpooled = torch.cat(
        [region01_mean, region2.repeat(1, 1, 2, 1)], dim=2
    )

    assert torch.allclose(x_unpooled, expected_unpooled, atol=1e-6)
    print("✓ test_dynamic_pool_unpool_mean_no_upsampling passed")


def test_dynamic_pool_unpool_mean_with_upsampling():
    """
    Dynamic branch: test UpS (temporal upsampling).
    Verify doubled temporal resolution.
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}
    x = make_dynamic_input()  # [B=1, T=2, N=3, C=4]

    pool = SkeletalPooling(adj, p=2, mode="mean", downsampling_params={"kernel_size": (1, 1)})
    pooled, regions, new_adj = pool(x)

    # Temporal upsampling: double time dimension
    unpool = SkeletalUnpooling(
        adj,
        regions,
        upsampling_params={"scale_factor": (1, 2), "mode": "nearest"},
    )
    x_unpooled = unpool(pooled)

    # Should have doubled temporal dimension (T=4)
    assert x_unpooled.shape[1] == 4
    print("✓ test_dynamic_pool_unpool_mean_with_upsampling passed")


def test_pool_unpool_roundtrip_preserves_shape():
    """
    Roundtrip shape consistency check for both branches.
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}

    x_static = make_static_input()
    pool_static = SkeletalPooling(adj, p=2)
    pooled_s, regions_s, _ = pool_static(x_static)
    unpool_static = SkeletalUnpooling(adj, regions_s)
    x_static_restored = unpool_static(pooled_s)
    assert x_static_restored.shape == x_static.shape

    x_dynamic = make_dynamic_input()
    pool_dynamic = SkeletalPooling(adj, p=2, downsampling_params={"kernel_size": (1, 1)})
    pooled_d, regions_d, _ = pool_dynamic(x_dynamic)
    unpool_dynamic = SkeletalUnpooling(adj, regions_d)
    x_dynamic_restored = unpool_dynamic(pooled_d)
    assert x_dynamic_restored.shape == x_dynamic.shape
    print("✓ test_pool_unpool_roundtrip_preserves_shape passed")


if __name__ == "__main__":
    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")

    test_static_pool_unpool_identity_mean()
    test_static_pool_unpool_max()
    test_dynamic_pool_unpool_mean_no_upsampling()
    test_dynamic_pool_unpool_mean_with_upsampling()
    test_pool_unpool_roundtrip_preserves_shape()
    print("\nAll pooling–unpooling integration tests passed successfully.")
