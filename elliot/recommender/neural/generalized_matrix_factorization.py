"""
Module description:

"""

import torch
from torch import nn

from elliot.dataset.samplers import PointWisePosNegSampler
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_uniform_init


class GMF(GeneralRecommender):
    r"""
    Generalized Matrix Factorization (GMF).

    For further details, please refer to the
    `paper <https://arxiv.org/abs/1708.05031>`_.

    Args:
        mf_factors: Number of latent factors
        lr: Learning rate
        is_edge_weight_train: Whether to train the edge-weight vector and use sigmoid + BCE
        batch_eval_items: Number of items per evaluation block

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        GMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          mf_factors: 10
          lr: 0.001
          is_edge_weight_train: True
    """

    mf_factors: int = 10
    lr: float = 0.001
    is_edge_weight_train: bool = True
    batch_eval_items: int = 256

    @property
    def name(self):
        return "GeneralizedMF"

    def __init__(self, data, params, seed, logger):
        super().__init__(data, params, seed, logger)

        self.user_mf_embedding = nn.Embedding(self._num_users, self.mf_factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.mf_factors, dtype=torch.float32)

        if self.is_edge_weight_train:
            self.edge_weight = nn.Parameter(torch.empty(self.mf_factors, 1, dtype=torch.float32))
            self.loss = nn.BCELoss()
        else:
            self.register_buffer("edge_weight", torch.ones(self.mf_factors, 1, dtype=torch.float32))
            self.loss = nn.MSELoss()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        self.apply(xavier_uniform_init)

        if self.is_edge_weight_train:
            nn.init.xavier_uniform_(self.edge_weight)

        self.to(self._device)

    def get_training_dataloader(self, batch_size):
        return self._data.training_dataloader(
            PointWisePosNegSampler,
            batch_size,
            self._seed,
        )

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        mf_output = user_mf_e * item_mf_e
        output = torch.matmul(mf_output, self.edge_weight).squeeze(-1)

        if self.is_edge_weight_train:
            output = torch.sigmoid(output)

        return output

    def train_step(self, batch, *args):
        user, item, label = [x.to(self._device) for x in batch]
        output = self.forward(user, item)
        return self.loss(output, label.float())

    def predict_full(self, user_indices):
        user_indices = user_indices.to(self._device)
        batch_size = len(user_indices)
        preds = []

        for s in range(0, self._num_items, self.batch_eval_items):
            e = min(s + self.batch_eval_items, self._num_items)
            items_block = torch.arange(s, e, device=self._device)

            users_block = user_indices.unsqueeze(1).expand(-1, e - s).reshape(-1)
            items_block_expanded = items_block.unsqueeze(0).expand(batch_size, -1).reshape(-1)

            preds_block = self.forward(users_block, items_block_expanded)
            preds.append(preds_block.view(batch_size, e - s))

        predictions = torch.cat(preds, dim=1)
        return predictions.to(self._device)

    def predict_sampled(self, user_indices, item_indices):
        user_indices = user_indices.to(self._device)
        item_indices = item_indices.to(self._device)
        batch_size, pad_seq = item_indices.size()

        users_expanded = user_indices.unsqueeze(1).expand(-1, pad_seq).reshape(-1)
        items_expanded = item_indices.clamp(min=0).reshape(-1)

        predictions_flat = self.forward(users_expanded, items_expanded)
        predictions = predictions_flat.view(batch_size, pad_seq)

        return predictions.to(self._device)
