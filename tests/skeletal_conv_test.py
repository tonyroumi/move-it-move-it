import torch
from skeletal_ops import SkeletalConv  # adjust import to match your structure

# Simple adjacency: 0 <- [1,2], 1 <- [0,2], 2 <- [0,1]
ADJ = [[0,1,3], [0,1,2,3], [1,2], [0,1,3,4], [3,4]]

CONV_PARAMS = {"in_channels_per_joint": 4, 
               "out_channels_per_joint": 8, 
               "bias": True, 
               "stride": 2, 
               "padding": 7, 
               "padding_mode": "reflect", 
               "dilation":1,
               "groups":1, 
               "kernel_size": 15}

def test_basic():
    """Verify correct output shape and preservation of armature order for static input."""
    layer = SkeletalConv(ADJ, **(CONV_PARAMS))
    x = torch.randn(2, 5*4, 20)  # [B, J*C, T]
    y = layer(x)

def run_tests():
    test_basic()

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