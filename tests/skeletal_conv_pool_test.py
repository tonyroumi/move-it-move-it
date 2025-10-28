import torch
from skeletal_ops import SkeletalPooling, SkeletalConv


def test_static_branch_pooling_mean():
    """
    Verify SkeletalPooling(mean) numerically after SkeletalConv(static).
    """
    # Simple linear chain: 0-1-2
    adj = {0: [1], 1: [0, 2], 2: [1]}

    # SkeletalConv setup: 3 inputs → 3 outputs (identity-like weights)
    conv_params = {"in_channels": 1, "out_channels": 1, "kernel_size": 1, "bias": False}
    conv = SkeletalConv(adj, conv_params)
    # Set each conv weight to 1 for deterministic output
    for layer in conv.support.values():
        layer.weight.data.fill_(1.0)

    # Input: one feature per node
    x = torch.tensor([
        [[1.0], [2.0], [3.0]]  # [B=1, J=3, C=1]
    ])

    conv_out = conv(x)  # Expected: averages over neighbors
    # Manually compute expected conv output
    # node0: mean([x1]) = 2
    # node1: mean([x0,x2]) = 2
    # node2: mean([x1]) = 2
    expected_conv = torch.tensor([[[2.0], [2.0], [2.0]]])
    assert torch.allclose(conv_out, expected_conv, atol=1e-6)

    # Pool: merge (0,1) and (2)
    pool = SkeletalPooling(adj, p=2, mode="mean")
    pooled, regions, _ = pool(conv_out)

    # Manually pooled: region [0,1] = mean([2,2])=2 ; region [2]=2
    expected_pool = torch.tensor([[[2.0], [2.0]]])
    assert torch.allclose(pooled, expected_pool, atol=1e-6)
    print("✓ test_static_branch_pooling_mean passed")


def test_static_branch_pooling_max():
    """
    Verify SkeletalPooling(max) after SkeletalConv(static).
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}
    conv_params = {"in_channels": 1, "out_channels": 1, "kernel_size": 1, "bias": False}
    conv = SkeletalConv(adj, conv_params)
    for layer in conv.support.values():
        layer.weight.data.fill_(1.0)

    x = torch.tensor([
        [[1.0], [2.0], [3.0]]
    ])
    conv_out = conv(x)
    pool = SkeletalPooling(adj, p=2, mode="max")
    pooled, regions, _ = pool(conv_out)

    # Max pooling: region [0,1]=max(2,2)=2 ; [2]=2
    expected_pool = torch.tensor([[[2.0], [2.0]]])
    assert torch.allclose(pooled, expected_pool, atol=1e-6)
    print("✓ test_static_branch_pooling_max passed")


def test_dynamic_branch_pooling_mean():
    """
    Verify SkeletalPooling(mean) on temporal data after SkeletalConv(dynamic).
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}
    conv_params = {"in_channels": 4, "out_channels": 1, "kernel_size": 1, "bias": False}
    conv = SkeletalConv(adj, conv_params)
    for layer in conv.support.values():
        layer.weight.data.fill_(1.0)

    # Input: [B=1, T=2, J=3, C=4]
    x = torch.arange(1, 25, dtype=torch.float32).view(1, 2, 3, 4)
    # Node features increase monotonically per node/time
    conv_out = conv(x)  # [B, T, J, Cout]
    pool = SkeletalPooling(adj, p=2, mode="mean", downsampling_params={"kernel_size": (1, 1)})
    pooled, regions, _ = pool(conv_out)

    # Manual expected pooling: each region mean across nodes.
    region0 = conv_out[:, :, [0, 1], :].mean(dim=2)
    region1 = conv_out[:, :, [2], :].mean(dim=2)
    expected = torch.stack([region0, region1], dim=2)
    assert torch.allclose(pooled, expected, atol=1e-6)
    print("✓ test_dynamic_branch_pooling_mean passed")


def test_dynamic_branch_pooling_max():
    """
    Verify SkeletalPooling(max) on temporal data after SkeletalConv(dynamic).
    """
    adj = {0: [1], 1: [0, 2], 2: [1]}
    conv_params = {"in_channels": 4, "out_channels": 1, "kernel_size": 1, "bias": False}
    conv = SkeletalConv(adj, conv_params)
    for layer in conv.support.values():
        layer.weight.data.fill_(1.0)

    x = torch.arange(1, 25, dtype=torch.float32).view(1, 2, 3, 4)
    conv_out = conv(x)
    pool = SkeletalPooling(adj, p=2, mode="max", downsampling_params={"kernel_size": (1, 1)})
    pooled, regions, _ = pool(conv_out)

    # Manual expected max pooling: across nodes per region.
    region0 = conv_out[:, :, [0, 1], :].max(dim=2).values
    region1 = conv_out[:, :, [2], :].max(dim=2).values
    expected = torch.stack([region0, region1], dim=2)
    assert torch.allclose(pooled, expected, atol=1e-6)
    print("✓ test_dynamic_branch_pooling_max passed")


if __name__ == "__main__":
    test_static_branch_pooling_mean()
    test_static_branch_pooling_max()
    test_dynamic_branch_pooling_mean()
    test_dynamic_branch_pooling_max()
    print("\nAll conv→pool integration tests passed successfully.")
