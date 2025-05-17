from math import log
import torch
import torch.nn as nn
from torch.nn import Module, Parameter, Dropout
import torch.nn.functional as F
from torch_geometric.nn import norm
from torch_geometric.nn.inits import glorot


class GCNLayer(Module):
    def __init__(self, channels, alpha, lamda=None, layer=None, shared_weights=True):
        super(GCNLayer, self).__init__()

        self.channels = channels
        self.alpha = alpha
        self.beta = 1.
        if lamda is not None or layer is not None:
            assert lamda is not None and layer is not None
            self.beta = log(lamda / layer + 1)
        self.weight1 = Parameter(torch.Tensor(channels, channels))
        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(channels, channels))

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight1)
        glorot(self.weight2)

    def forward(self, x, x_0, adj):
        x = torch.matmul(adj, x)
        if self.weight2 is None:
            out = (1 - self.alpha) * x + self.alpha * x_0
            out = (1 - self.beta) * out + self.beta * (out @ self.weight1)
        else:
            out1 = (1 - self.alpha) * x
            out1 = (1 - self.beta) * out1 + self.beta * (out1 @ self.weight1)
            out2 = self.alpha * x_0
            out2 = (1 - self.beta) * out2 + self.beta * (out2 @ self.weight2)
            out = out1 + out2
        return out


class DeepGCNLayer(Module):
    def __init__(self, channels, alpha=0.5, lamda=0.5, layer=None, dropout=0.5):
        super(DeepGCNLayer, self).__init__()
        self.gcn = GCNLayer(channels, alpha=alpha, lamda=lamda, layer=layer)
        self.bn = norm.BatchNorm(channels)
        self.dropout = Dropout(p=dropout)
        self.act_fn = nn.ReLU()

    def forward(self, x, x_0, adj):
        hidden = self.gcn(x, x_0, adj)
        hidden = torch.squeeze(self.bn(hidden))
        hidden = self.act_fn(hidden)
        hidden = self.dropout(hidden)
        return hidden
