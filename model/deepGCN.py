import torch as t
from torch import nn
from torch_geometric.nn import norm
from model.layers import DeepGCNLayer


class DeepGCN(nn.Module):
    def __init__(self, nfeat, nclass, dim=512, alpha=0.5, lamda=0.5, dropout=0.3):
        super(DeepGCN, self).__init__()

        self.nfeat = nfeat
        self.nclass = nclass

        self.bn_in = norm.BatchNorm(nfeat)
        self.act_fn = nn.ReLU()
        self.linear_in = nn.Linear(nfeat, dim, bias=True)

        self.gcn1 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=1, dropout=dropout)
        self.gcn2 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=2, dropout=dropout)
        self.gcn3 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=3, dropout=dropout)
        self.gcn4 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=4, dropout=dropout)
        self.gcn5 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=5, dropout=dropout)
        self.gcn6 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=6, dropout=dropout)
        self.gcn7 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=7, dropout=dropout)
        self.gcn8 = DeepGCNLayer(dim, alpha=alpha, lamda=lamda, layer=8, dropout=dropout)

        self.dropout = nn.Dropout(p=dropout)
        self.classify = nn.Linear(dim, nclass)

    def forward(self, x, net):
        x = t.squeeze(self.bn_in(x))
        x = self.act_fn(self.linear_in(x))

        hidden = self.gcn1(x, x, net)
        hidden = self.gcn2(hidden, x, net)
        hidden = self.gcn3(hidden, x, net)
        hidden = self.gcn4(hidden, x, net)
        hidden = self.gcn5(hidden, x, net)
        hidden = self.gcn6(hidden, x, net)
        hidden = self.gcn7(hidden, x, net)
        hidden = self.gcn8(hidden, x, net)

        emb = self.act_fn(hidden)
        emb = self.dropout(emb)
        pred = self.classify(emb)
        return pred
