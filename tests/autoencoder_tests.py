from models import SkeletalAutoEncoder, SkeletalEncoder
from tests.utils import load_params, ADJ, EDGE_LIST, BATCH_SIZE, WINDOW_SIZE
from typing import Dict, List
import numpy as np
import torch

def simple_test():
    decoder = load_params("configs/default/decoder.yaml")
    dynamic_params = load_params("configs/default/dynamic_encoder.yaml")
    static_params = load_params("configs/default/static_encoder.yaml")
    params = {"decoder": decoder, "encoder": dynamic_params}

    offsets = torch.rand((1, len(ADJ)*3))
    dynamic_input = torch.rand((BATCH_SIZE, len(ADJ)*4, WINDOW_SIZE))

    static_encoder = SkeletalEncoder(adj_init=ADJ, 
                                    edge_init=EDGE_LIST, 
                                    encoder_params=static_params)
    
    offset_features, all_offset_features = static_encoder(offsets.unsqueeze(-1))

    auto_encoder = SkeletalAutoEncoder(ADJ, EDGE_LIST, params)

    auto_encoder(dynamic_input, all_offset_features)
    
#1st unpool will be second unpoling list, 16 channels_per_edge, 1st pooled neighbor_list, joint_num: 12
#2nd unpool will be original unpooling list, 8 channels_per_edge, original neighbor_list, joint_num: 23

if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")

    simple_test()

    
    