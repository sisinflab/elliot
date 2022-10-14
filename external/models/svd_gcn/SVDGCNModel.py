from abc import ABC

import torch
import numpy as np
import random


class SVDGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 learning_rate,
                 embed_k,
                 l_w,
                 coef_u,
                 coef_i,
                 beta,
                 req_vec,
                 u,
                 value,
                 v,
                 random_seed,
                 name="SVDGCN",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.l_w = l_w
        self.coef_u = coef_u
        self.coef_i = coef_i
        self.beta = beta

        svd_filter = self.weight_func(value[:req_vec].to(self.device))
        self.user_vector = (u[:, :req_vec]).to(self.device) * svd_filter
        self.item_vector = (v[:, :req_vec]).to(self.device) * svd_filter
        self.FS = torch.nn.Parameter(
            torch.nn.init.uniform_(torch.randn(req_vec, self.embed_k), -np.sqrt(6. / (req_vec + self.embed_k)),
                                   np.sqrt(6. / (req_vec + self.embed_k))).to(self.device))

    def weight_func(self, sig):
        return torch.exp(self.beta * sig)

    def forward(self, inputs, **kwargs):
        emb1, emb2 = inputs
        emb1_final = torch.squeeze(emb1).to(self.device).mm(self.FS)
        emb2_final = torch.squeeze(emb2).to(self.device).mm(self.FS)

        out = torch.sum(emb1_final * emb2_final, 1)

        return emb1_final, emb2_final, out

    def train_step(self, batch):
        u, p, n, up, un, pp, pn = batch

        final_user, final_pos, out_ui_pos = self.forward(inputs=(self.user_vector[u], self.item_vector[p]))
        _, final_nega, out_ui_neg = self.forward(inputs=(self.user_vector[u], self.item_vector[n]))
        _, final_user_p, out_uu_pos = self.forward(inputs=(self.user_vector[u], self.user_vector[up]))
        _, final_user_n, out_uu_neg = self.forward(inputs=(self.user_vector[u], self.user_vector[un]))
        _, final_pos_p, out_ii_pos = self.forward(inputs=(self.item_vector[p], self.item_vector[pp]))
        _, final_pos_n, out_ii_neg = self.forward(inputs=(self.item_vector[p], self.item_vector[pn]))

        loss_ui = torch.log((out_ui_pos - out_ui_neg).sigmoid()).sum()
        loss_uu = torch.log((out_uu_pos - out_uu_neg).sigmoid()).sum()
        loss_ii = torch.log((out_ii_pos - out_ii_neg).sigmoid()).sum()

        regu_term = self.l_w * (
                    final_user ** 2 + final_pos ** 2 + final_nega ** 2 + final_user_p ** 2 + final_user_n ** 2 + final_pos_p ** 2 + final_pos_n ** 2).sum()

        loss = (-loss_ui-self.coef_u*loss_uu-self.coef_i*loss_ii+regu_term)/u.shape[0]
        loss.backward()
        with torch.no_grad():
            self.FS -= self.learning_rate * self.FS.grad
            self.FS.grad.zero_()

        return loss.detach().cpu().numpy()

    def get_top_k(self, preds, train_mask, k=100):
        return torch.topk(torch.where(torch.tensor(train_mask).to(self.device), preds.to(self.device),
                                      torch.tensor(-np.inf).to(self.device)), k=k, sorted=True)
