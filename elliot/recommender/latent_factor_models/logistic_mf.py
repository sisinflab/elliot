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
class LogisticMF(GeneralRecommender):
    """
    Logistic Matrix Factorization

    For further details, please refer to the `paper <https://web.stanford.edu/~rezab/nips2014workshop/submits/logmat.pdf>`_

    Args:
        factors: Number of factors of feature embeddings
        lr: Learning rate
        reg: Regularization coefficient
        alpha: Parameter for confidence estimation

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        LogisticMatrixFactorization:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          learning_rate: 0.001
          lambda_weights: 0.1
          alpha: 0.5
    """

    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.1
    alpha: float = 0.5

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super(LogisticMF, self).__init__(params, seed, interactions, *args, **kwargs)

        # Embeddings
        self.Gu = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.Gi = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)
        self.Bu = nn.Embedding(self._num_users, 1, dtype=torch.float32)
        self.Bi = nn.Embedding(self._num_items, 1, dtype=torch.float32)

        # Optimizer
        # NOTE: Removed Adagrad optimizer due to its poor performance
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        self.transactions = self._interactions.transactions * 2

        # Sampler configuration
        self.sampler_config = {
            "name": "PointWisePosNegSampler",
            "transactions": self.transactions
        }

        # Init embedding weights
        self.bias = [self.Bu, self.Bi]
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def forward(self, user, item):
        user_e = self.Gu(user)
        item_e = self.Gi(item)
        user_b = self.Bu(user)
        item_b = self.Bi(item)

        xui = torch.mul(user_e, item_e).sum(dim=1) + user_b + item_b
        return xui

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]
        label = label.float()

        output = self.forward(user, pos)

        reg = self.Gu.weight[user].pow(2).sum() + self.Gi.weight[pos].pow(2).sum()
        loss = torch.sum(
            - (self.alpha * label * output - (1 + self.alpha * label) * torch.log1p(torch.exp(output)))
        ) + self.lambda_weights * reg

        steps = args[0]
        inputs = ([self.Gu.weight, self.Bu.weight] if steps > self._interactions.transactions
                  else [self.Gi.weight, self.Bi.weight])

        return loss, inputs

    def predict(self, user_indices, item_indices=None, **kwargs):
        user_e_all = self.Gu.weight
        item_e_all = self.Gi.weight
        user_b_all = self.Bu.weight
        item_b_all = self.Bi.weight

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
