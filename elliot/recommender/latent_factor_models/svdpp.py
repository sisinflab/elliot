"""
Module description:

"""


import torch
from torch import nn
from torch_sparse import SparseTensor

from elliot.dataset import Interactions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_uniform_init
from elliot.utils.registry import model_registry


@model_registry.register()
class SVDpp(GeneralRecommender):
    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1
    lambda_bias: float = 0.001

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super(SVDpp, self).__init__(params, seed, interactions, *args, **kwargs)

        # Row-sliceable (users x items) interaction matrix, for `_compute_user_representation`
        interact_coo = self._interactions.sparse_tensor.coalesce()
        row, col = interact_coo.indices()
        self.interact_mat = SparseTensor(row=row, col=col, sparse_sizes=interact_coo.shape).to(self._device)

        # Embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.item_y_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.user_bias_embedding = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.item_bias_embedding = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Global bias
        self.bias_ = nn.Parameter(torch.zeros(1))

        # Loss and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Sampler configuration
        self.sampler_config = {
            "name": "CustomPointWiseSparseSampler"
        }

        # Init embedding weights
        self.bias = [self.user_bias_embedding, self.item_bias_embedding]
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
        u = self.user_mf_embedding(user)
        i = self.item_mf_embedding(item)
        ub = self.user_bias_embedding(user)
        ib = self.item_bias_embedding(item)

        puyj = self._compute_user_representation(user)

        output = torch.mul((puyj + u), i).sum(dim=-1) + ub.squeeze() + ib.squeeze() + self.bias_
        return output

    def train_step(self, batch, *args):
        user, item, label = [x.to(self._device) for x in batch]

        output = self.forward(user, item)

        reg = (
            self.lambda_weights * (
                self.user_mf_embedding.weight.pow(2).sum() +
                self.item_mf_embedding.weight.pow(2).sum() +
                self.item_y_embedding.weight.pow(2).sum()
            ) +
            self.lambda_bias * (
                self.user_bias_embedding.weight.pow(2).sum() +
                self.item_bias_embedding.weight.pow(2).sum()
            )
        )
        loss = self.loss(label.float(), output) + reg

        return loss

    def predict(self, user_indices, item_indices=None, **kwargs):
        user_e_all = self.user_mf_embedding.weight
        item_e_all = self.item_mf_embedding.weight
        user_b_all = self.user_bias_embedding.weight
        item_b_all = self.item_bias_embedding.weight

        # Select only the embeddings in the current batch
        user_embeddings = user_e_all[user_indices]
        user_bias = user_b_all[user_indices]

        # Compute predictions
        puyj = self._compute_user_representation(user_indices)

        if item_indices is None:
            item_embeddings = item_e_all
            item_bias = item_b_all.T
            einsum_string = "be,ie->bi"
        else:
            item_indices = item_indices.clamp(min=0)
            item_embeddings = item_e_all[item_indices]
            item_bias = item_b_all[item_indices].squeeze(-1)
            einsum_string = "be,bse->bs"

        predictions = (
            torch.einsum(
                einsum_string, (puyj + user_embeddings), item_embeddings
            )
            + user_bias
            + item_bias
            + self.bias_
        )

        return predictions

    def _compute_user_representation(self, users):
        item_y_all = self.item_y_embedding.weight
        offsets, indices, _ = self.interact_mat[users].csr()

        puyj = nn.functional.embedding_bag(
            input=indices,
            weight=item_y_all,
            offsets=offsets[:-1],
            mode='mean'
        )

        return puyj
