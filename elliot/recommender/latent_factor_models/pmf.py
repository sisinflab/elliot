"""
Module description:

Mnih, Andriy, and Russ R. Salakhutdinov. "Probabilistic matrix factorization." Advances in neural information processing systems 20 (2007)

"""


import torch
from torch import nn

from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.recommender.layers import GaussianNoise
from elliot.utils.registry import model_registry


@model_registry.register()
class PMF(GeneralRecommender):
    """
    Probabilistic Matrix Factorization

    For further details, please refer to the `paper <https://papers.nips.cc/paper/2007/file/d7322ed717dedf1eb4e6e52a37ea7bcd-Paper.pdf>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        reg: Regularization coefficient
        gaussian_variance: Variance of the Gaussian distribution

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        PMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 50
          learning_rate: 0.001
          lambda_weights: 0.0025
          gaussian_variance: 0.1
    """

    # Model hyperparameters
    factors: int = 10
    learning_rate: float = 0.001
    lambda_weights: float = 0.0025
    gaussian_variance: float = 0.1

    def __init__(self, params, interactions, seed, *args, **kwargs):
        super(PMF, self).__init__(params, interactions, seed, *args, **kwargs)

        # Embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)

        # Gaussian noise
        self.noise = GaussianNoise(self.gaussian_variance)

        # Activation function, loss and optimizer
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def get_training_dataloader(self, batch_size):
        dataloader = self._interactions.get_dataloader("PointWisePosNegSampler", batch_size, self._seed)
        return dataloader

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)

        mf_output = torch.mul(user_mf_e, item_mf_e).sum(dim=1)
        output = self.sigmoid(mf_output)
        return output

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]

        output = self.noise(self.forward(user, pos))

        reg = self.user_mf_embedding.weight.pow(2).sum() + self.item_mf_embedding.weight.pow(2).sum()
        loss = self.loss(label.float(), output) + self.lambda_weights * reg

        return loss

    def predict(self, user_indices, item_indices=None):
        user_e_all = self.user_mf_embedding.weight
        item_e_all = self.item_mf_embedding.weight

        # Select only the embeddings in the current batch
        user_embeddings = user_e_all[user_indices]

        # Compute predictions
        if item_indices is None:
            item_embeddings = item_e_all
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = item_e_all[item_indices.clamp(min=0)]
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(
            einsum_string, user_embeddings, item_embeddings
        )
        predictions = self.sigmoid(predictions)

        return predictions
