from abc import ABC

import torch


class FeatureProjection(torch.nn.Module, ABC):
    def __init__(self, in_channels, out_channels, disen_k):
        super(FeatureProjection, self).__init__()
        self.W = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((disen_k, in_channels, out_channels // disen_k))))
        self.b = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((disen_k, out_channels // disen_k))))
        self.relu = torch.nn.ReLU()
        self.disen_k = disen_k

    def forward(self, x):
        z = self.relu(torch.matmul(self.W.permute(0, 2, 1), x.permute(1, 0)).permute(2, 0, 1) +
                      torch.unsqueeze(self.b, 0))
        z = z / torch.unsqueeze(torch.unsqueeze(torch.norm(z, 2, dim=[0, 2]), 0), 2)

        return z
