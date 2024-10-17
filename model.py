import torch.nn as nn
import torch
import math
import torch.nn.functional as F
# import dgl.function as fn
# from gcn import GCN
from mamba_ssm import Mamba
class ClassMLP(torch.nn.Module):
    def __init__(self, snapshot, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super(ClassMLP, self).__init__()
        self.mamba =  Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=in_channels, # Model dimension d_model
        d_state=1,  # SSM state expansion factor
        d_conv=4,    # Local convolution width
        expand=2,    # Block expansion factor
        ).to("cuda")
        self.snapshot = snapshot
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        if(self.snapshot>0):
            x = self.mamba(x)
            x = x[:, -1, :]
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return torch.log_softmax(x, dim=-1)
        # return x


