from typing import Dict, List, Literal
import torch
import torch.nn as nn

class SkeletalConv(nn.Module):
    """
    Skeletal-temporal convolution.

    Args
    ----
    adj_list      :   Dict[int, List[int]]
                      N_i^d neighbors per edge i.

    conv_params   :   Dict[str, int]
                      Convolution parameters.
    """

    def __init__(
        self,
        adj_list: Dict[int, List[int]],
        conv_params: Dict[str, int]
    ):
        super().__init__()
        self.adj = adj_list
        self.edges = sorted(self.adj.keys())
        self.E = len(self.edges)

        self.conv_params = conv_params
        
        self._init_net()

    def _init_net(self):
        support = {}
        for i in self.edges:
            for j in self.adj[i]:
                key = f"{i}<-{j}"
                support[key] = nn.Conv1d(
                    **(self.conv_params)
                )
        self.support = nn.ModuleDict(support)
    
    def forward(self, x: torch.Tensor):
        """
        The input to the network should be a matrix where each slice along the second dim corresponds to the same
        edge as specified in self.adj.

        For static inputs:
        B, J, 3

        For dynamic inputs:
        B, T, J, 4
        """
        #Static branch
        if x.dim() == 3:
            y_stat_out = []
            for edge in self.edges:
                acc = 0
                deg = len(self.adj[edge])
                for neighbor in self.adj[edge]:
                    key = f"{edge}<-{neighbor}"
                    out_ij = self.support[key](x[:, neighbor, :])
                    acc += out_ij 
                acc /= deg                                          # 1 / |N_i^d|
                y_stat_out.append(acc)
            y_stat_feat = torch.stack(y_stat_out, dim=1)            # [B, J, Cout]
            return y_stat_feat

        #Dynamic branch
        else:
            y_dyn_out = []
            for edge in self.edges:
                acc = 0
                deg = len(self.adj[edge])
                for neighbor in self.adj[edge]:
                    key = f"{edge}<-{neighbor}"
                    xj = x[:, :, neighbor, :].transpose(1,2)        # [B, 4, T]
                    out_ij = self.dynamic[key](xj)
                    acc += out_ij
                acc /= deg                                          # 1 / |N_i^d|
                y_dyn_out.append(acc.transpose(1,2))                # [B, T, Cout]
            y_dyn_feat = torch.stack(y_dyn_out, dim=2)              # [B, T, J, Cout]
            return y_dyn_feat
