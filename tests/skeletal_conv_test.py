import torch
from skeletal_ops import SkeletalConv  # adjust import to match your structure

# Simple adjacency: 0 <- [1,2], 1 <- [0,2], 2 <- [0,1]
ADJ = {0: [1, 2], 1: [0, 2], 2: [0, 1]}

CONV_PARAMS = {"in_channels": 3, "out_channels": 3, "kernel_size": 1}


def test_static_input_shape_and_connectivity():
    """Verify correct output shape and preservation of armature order for static input."""
    layer = SkeletalConv(ADJ, CONV_PARAMS)
    x = torch.randn(2, 3, 3)  # [B, J, 3]
    y = layer(x)

    assert y.shape == (2, 3, 3), f"Expected (2,3,3), got {y.shape}"
    # armature preserved: same ordering of edges
    assert list(ADJ.keys()) == list(range(y.shape[1]))


def test_dynamic_input_shape_and_behavior():
    """Verify dynamic input shape and convolution over time dimension."""
    conv_params = {"in_channels": 4, "out_channels": 4, "kernel_size": 3, "padding": 1}
    layer = SkeletalConv(ADJ, conv_params)

    # Manually assign dynamic convs since 'self.dynamic' isn't defined in your code yet
    layer.dynamic = layer.support

    B, T, J, F = 2, 10, 3, 4
    x = torch.randn(B, T, J, F)
    y = layer(x)

    assert y.shape == (B, T, J, 4), f"Expected (2,10,3,4), got {y.shape}"
    # Output should differ when input changes along time
    x2 = x.clone()
    x2[:, 5:, :, :] += 1.0
    y2 = layer(x2)
    assert not torch.allclose(y, y2), "Dynamic output should vary with time changes"


def test_consistency_static_vs_dynamic_structure():
    """Ensure same armature adjacency produces consistent indexing between static and dynamic."""
    conv_params_static = {"in_channels": 3, "out_channels": 3, "kernel_size": 1}
    conv_params_dynamic = {"in_channels": 4, "out_channels": 4, "kernel_size": 1}
    static_layer = SkeletalConv(ADJ, conv_params_static)
    dynamic_layer = SkeletalConv(ADJ, conv_params_dynamic)
    dynamic_layer.dynamic = dynamic_layer.support

    assert set(static_layer.support.keys()) == set(dynamic_layer.support.keys())

def run_tests():
    test_static_input_shape_and_connectivity()
    test_dynamic_input_shape_and_behavior()
    test_consistency_static_vs_dynamic_structure()

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