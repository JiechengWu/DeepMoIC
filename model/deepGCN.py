import torch as t
from torch import nn
from torch_geometric.nn import norm
from model.layers import DeepGCNLayer


class DeepGCN(nn.Module):
    def __init__(self, nfeat, nclass, dim=512, alpha=0.5, lamda=0.5, nlayers=2, dropout=0.3):
        super(DeepGCN, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass

        self.bn_in = norm.BatchNorm(nfeat)
        self.act_fn = nn.ReLU()
        self.linear_in = nn.Linear(nfeat, dim, bias=True)
        self.convs = nn.ModuleList()

        for i in range(nlayers):
            self.convs.append(DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=i + 1, dropout=dropout))

        self.dropout = nn.Dropout(p=dropout)
        self.classify = nn.Linear(dim, nclass)

    def forward(self, x, net):
        x = t.squeeze(self.bn_in(x))
        x = self.act_fn(self.linear_in(x))

        for i, con in enumerate(self.convs):
            if i == 0:
                hidden = con(x, x, net)
            else:
                hidden = con(hidden, x, net)

        emb = self.act_fn(hidden)
        emb = self.dropout(emb)
        pred = self.classify(emb)
        return pred
