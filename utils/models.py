import torch.nn as nn
import torch.nn.functional as F
from utils.layers import GraphConvolution

"""
src: https://github.com/tkipf/pygcn
"""
class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        # return x
        return F.log_softmax(x, dim=1)



class MLP(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(MLP, self).__init__()
        self.lay1 = nn.Linear(nfeat, nhid)
        self.lay2 = nn.Linear(nhid, nout)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.lay1(x))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lay2(x)
        # return x
        return F.log_softmax(x, dim=1)