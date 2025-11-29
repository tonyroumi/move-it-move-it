from src.skeletal_ops import SkeletalPooling
from typing import Dict, List
import numpy as np
import torch

ADJ = [
    [0, 1, 2, 4, 5, 8, 9, 22],
    [0, 1, 2, 3, 4, 8, 22],
    [0, 1, 2, 3, 22],
    [1, 2, 3],
    [0, 1, 4, 5, 6, 8, 9, 22],
    [0, 4, 5, 6, 7, 8, 22],
    [4, 5, 6, 7],
    [5, 6, 7],
    [0, 1, 4, 5, 8, 9, 10, 22],
    [0, 4, 8, 9, 10, 11, 22],
    [8, 9, 10, 11, 12, 14, 18],
    [9, 10, 11, 12, 13, 14, 15, 18, 19],
    [10, 11, 12, 13, 14, 15, 18, 19],
    [11, 12, 13, 14, 18],
    [10, 11, 12, 13, 14, 15, 16, 18, 19],
    [11, 12, 14, 15, 16, 17, 18],
    [14, 15, 16, 17],
    [15, 16, 17],
    [10, 11, 12, 13, 14, 15, 18, 19, 20],
    [11, 12, 14, 18, 19, 20, 21],
    [18, 19, 20, 21],
    [19, 20, 21],
    [0, 1, 2, 4, 5, 8, 9],
]

EDGE_LIST = [
    (np.int64(0), 1),
    (np.int64(1), 2),
    (np.int64(2), 3),
    (np.int64(3), 4),
    (np.int64(0), 5),
    (np.int64(5), 6),
    (np.int64(6), 7),
    (np.int64(7), 8),
    (np.int64(0), 9),
    (np.int64(9), 10),
    (np.int64(10), 11),
    (np.int64(11), 12),
    (np.int64(12), 13),
    (np.int64(13), 14),
    (np.int64(12), 15),
    (np.int64(15), 16),
    (np.int64(16), 17),
    (np.int64(17), 18),
    (np.int64(12), 19),
    (np.int64(19), 20),
    (np.int64(20), 21),
    (np.int64(21), 22),
]

def simple_test():
    pool = SkeletalPooling(EDGE_LIST, channels_per_edge=8)


    # Expect 3 pooled regions of length 2


def run_tests():
    simple_test()

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
