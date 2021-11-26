from abc import ABC

import torch


class AttentionPropagationLayer(torch.nn.Module, ABC):
    def __init__(self, embed_k):
        super(AttentionPropagationLayer, self).__init__()
        self.W = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((embed_k * 2, 1))))
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=0)

    def forward(self, x):
        all_x0_xk = torch.cat([torch.unsqueeze(torch.cat((x[0], x[i]), dim=1), dim=0) for i in range(x.shape[0])],
                              dim=0)
        betas = torch.squeeze(self.softmax(torch.matmul(all_x0_xk, self.W)))
        return betas.permute((1, 0))
