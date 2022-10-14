from abc import ABC

import torch
import numpy as np


class SVDGCNSModel(torch.nn.Module, ABC):
    def __init__(self,
                 beta,
                 req_vec,
                 u,
                 value,
                 v,
                 name="SVDGCNS",
                 **kwargs
                 ):
        super().__init__()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.beta = beta

        svd_filter = self.weight_func(value[:req_vec].to(self.device))
        self.user_vector = (u[:, :req_vec]).to(self.device) * svd_filter
        self.item_vector = (v[:, :req_vec]).to(self.device) * svd_filter

    def weight_func(self, sig):
        return torch.exp(self.beta * sig)

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
