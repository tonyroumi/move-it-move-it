from models import SkeletalDiscriminator, SkeletalEncoder
from tests.utils import load_params, ADJ, EDGE_LIST, BATCH_SIZE, WINDOW_SIZE
from typing import Dict, List
import numpy as np
import torch

def simple_test():
    discriminator_params = load_params("configs/default/discriminator.yaml")
    encoder_params = load_params("configs/default/static_encoder.yaml")

    fake_pos = torch.rand((1, 64, len(ADJ)*3))

    pooled_info = SkeletalEncoder(ADJ, EDGE_LIST, encoder_params=encoder_params).pooling_hierarchy

    disciminator = SkeletalDiscriminator(pooled_info, 
                                           discriminator_params)
    
    features, all_features = disciminator(fake_pos.permute(0,2,1))

if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")

    simple_test()

    
    