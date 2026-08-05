"""
Module description:

"""


import torch
from torch import nn

from elliot.dataset import Interactions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_uniform_init
from elliot.utils.registry import model_registry


@model_registry.register()
class FunkSVD(GeneralRecommender):
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
        super(FunkSVD, self).__init__(params, seed, interactions, *args, **kwargs)

        # Embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.user_bias_embedding = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.item_bias_embedding = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Loss and optimizer
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)

        # Sampler configuration
        self.sampler_config = {
            "name": "PointWisePosNegSampler"
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

        return torch.mul(u, i).sum(dim=1) + ub.squeeze() + ib.squeeze()

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]

        output = self.forward(user, pos)

        reg = (
            self.lambda_weights * (
                self.user_mf_embedding.weight.pow(2).sum() +
                self.item_mf_embedding.weight.pow(2).sum()
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
                einsum_string, user_embeddings, item_embeddings
            )
            + user_bias
            + item_bias
        )

        return predictions
