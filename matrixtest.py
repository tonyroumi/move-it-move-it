import torch
import torch.nn as nn
import torch.nn.functional as F

weight = torch.rand(10,10,1)

mask = torch.zeros(10,10,1)
expanded = [[0,1,2,3,6,7],
            [0,1,2,3,4,5,6,7],
            [2,3,4,5],
            [0,1,2,3,6,7,8,9],
            [6,7,8,9]]
for i, neighbor in enumerate(expanded):
    row_start = 2*i
    row_end = 2*(i+1)
    mask[row_start:row_end, neighbor, ...] = 1

weight_masked = weight * mask

print(weight_masked)