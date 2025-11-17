from models import SkeletalEncoder
from tests.utils import load_params, ADJ, EDGE_LIST, BATCH_SIZE, WINDOW_SIZE
from typing import Dict, List
import numpy as np
import torch

def simple_test():
    static_params = load_params("configs/default/static_encoder.yaml")
    dynamic_params = load_params("configs/default/dynamic_encoder.yaml")

    offsets = torch.rand((1, len(ADJ)*3))
    dynamic_input = torch.rand((BATCH_SIZE, len(ADJ)*4, WINDOW_SIZE))

    static_encoder = SkeletalEncoder(adj_init=ADJ, 
                                    edge_init=EDGE_LIST, 
                                    encoder_params=static_params, 
                                    type='static')
    
    dynamic_encoder = SkeletalEncoder(adj_init=ADJ,
                                      edge_init=EDGE_LIST,
                                      encoder_params=dynamic_params,
                                      type="dynamic")
    
    offset_features, offset_features = static_encoder(offsets.unsqueeze(-1))

    dynamic_features, _ = dynamic_encoder(dynamic_input, offset_features)

if __name__ == "__main__":

    import sys

    if "debug" in sys.argv:
        import debugpy
        print("[DEBUG] Waiting for debugger to attach on 0.0.0.0:5678 ...")
        debugpy.listen(("0.0.0.0", 5678))
        debugpy.wait_for_client()
        print("[DEBUG] Debugger attached.")

    simple_test()

    
    
# We have our offsets of shape (4, len(ADJ), 3)
#Pass these through the static encoder
# The static encoder enumerates through layers like so:
## 1.) Block1:
#### a.) SkeletalLinear(in_channels=3*num_joints (23 for them), out_channels = 6*num_joints)
#### Out of this comes (4, 138, 1) 
#### b.) SkeletalPool(contains pooling_list, new_edges, seq_list)
#### Out of this comes (4, 72, 1) (features were pooled)
#### c.) LeakyRELU

# they append all intermediate outputs.

## 2.) Block2:
#### a.) SkeletalLinear(in_channels=6*(NEW)num_joints (12 for them), out_channels = 12*num_joints)
#### Out of this comes (4, 144, 1) 
#### c.) LeakyRELU

## So the output they return is of len(output)=3. Before anything, after one block, after second block. 
## Respectively...
## (4,69) , (4, 72), (4, 144)

#NOTE THEY ONLY USE THE FIRST TWO. 

# They store the offsets so that in the dynamic encoder they can set it at each layer. 

# The dynamic encoder (only focused on encoder part) enumerates through layers like so:
# The forward of the encoder pads one zero row to the global position so each joint including global position has 4 channels. Not totally sure why yet.
## 1.) Block1:
## So they set the offset before a forward pass of an entire layer. The layers look like:

#set offset

# Layer(0)
## SkeletonConv(in_channels=4*num_joints (23 for them), out_channels = 8*num_joints)  OUT(B, 184, 32)
#####SkeletalLinear(in_channels=3*23, out_channels=8*23) WE want to project it onto the conv. 
##### Then they reshape it and output += offset_output / 100 in conv.
## SkeletonPool() OUT(B, 96, 32)
## LeakyRELU()

#set offset

# Layer(1)
## SkeletonConv(in_channels=8*num_joints (12 for them), out_channels = 16*num_joints)  OUT(B, 192, 16)
#####SkeletalLinear(in_channels=6*12, out_channels=16*12)
## SkeletonPool() OUT(B,112,16)
## LeakyRELU()