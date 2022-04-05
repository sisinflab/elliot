from abc import ABC

import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from torch_sparse import matmul, mul_nnz, sum
from torch_geometric.nn.inits import uniform


class MGATLayer(MessagePassing, ABC):
    def __init__(self, in_dim, out_dim, ptr, rows, cols, bias=True, normalize=True, aggr='add'):
        super(MGATLayer, self).__init__(aggr=aggr)
        self.ptr = ptr
        self.in_dim = in_dim
        self.out_dim = in_dim
        self.weight = torch.nn.Parameter(torch.Tensor(in_dim, out_dim))
        self.normalize = normalize
        self.rows = rows
        self.cols = cols
        uniform(in_dim, self.weight)
        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_dim))
            uniform(in_dim, self.bias)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, edge_index):
        # Compute attention coefficients.
        x = x.unsqueeze(-1) if x.dim() == 1 else x
        x = torch.matmul(x, self.weight)
        x_i = x[torch.cat((self.rows, self.cols), dim=0)].view(-1, self.out_dim)
        x_j = x[torch.cat((self.cols, self.rows), dim=0)].view(-1, self.out_dim)
        inner_product = torch.mul(x_i, self.leaky_relu(x_j)).sum(dim=-1)

        # gate
        deg = sum(edge_index, dim=-1)
        deg_inv_sqrt = deg[torch.cat((self.rows, self.cols), dim=0)].pow_(-0.5)
        tmp = torch.mul(deg_inv_sqrt, inner_product)
        gate_w = torch.sigmoid(tmp)

        # attention
        tmp = torch.mul(inner_product, gate_w)
        try:
            attention_w = softmax(src=tmp, ptr=self.ptr).view(-1)
        except:
            print(tmp)
            print(self.ptr)
            exit()
        edge_index = mul_nnz(edge_index, attention_w, layout='coo')

        return self.propagate(edge_index, x=x)

    def update(self, aggr_out):
        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        if self.normalize:
            aggr_out = torch.nn.functional.normalize(aggr_out, p=2, dim=-1)
        return aggr_out

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)
