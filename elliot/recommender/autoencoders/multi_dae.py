"""
Collaborative Denoising AutoEncoder (PyTorch, GeneralRecommender).

"""


import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from elliot.dataset import Interactions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.autoencoders.layers import DenoisingAutoEncoder
from elliot.recommender.init import xavier_normal_init
from elliot.utils.registry import model_registry


@model_registry.register()
class MultiDAE(GeneralRecommender):
    r"""
    Collaborative denoising autoencoder

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3178876.3186150>`_

    Args:
        intermediate_dim: Number of intermediate dimension
        latent_dim: Number of latent factors
        reg_lambda: Regularization coefficient
        lr: Learning rate
        dropout_pkeep: Dropout probability to keep (1 means no dropout)

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        MultiDAE:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          intermediate_dim: 600
          latent_dim: 200
          reg_lambda: 0.01
          lr: 0.001
          dropout_pkeep: 1
    """

    # Model hyperparameters
    intermediate_dim: int = 600
    latent_dim: int = 200
    reg_lambda: float = 0.01
    lr: float = 0.001
    dropout_pkeep: float = 1.0

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)

        dropout_rate = max(0.0, 1.0 - self.dropout_pkeep)

        self._train_matrix = self._interactions.sparse
        self._autoencoder = DenoisingAutoEncoder(
            original_dim=self._num_items,
            intermediate_dim=self.intermediate_dim,
            latent_dim=self.latent_dim,
            dropout_rate=dropout_rate,
        )

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.transactions = self._num_users

        self.sampler_config = {
            "name": "SparseSampler",
            "sparse": self._train_matrix
        }

        self.apply(xavier_normal_init)
        self.to(self._device)

    def train_step(self, batch, *args):
        batch = batch.to(self._device)
        logits = self._autoencoder(batch)
        log_softmax_var = F.log_softmax(logits, dim=1)

        neg_ll = -torch.mean(torch.sum(log_softmax_var * batch, dim=1))
        loss = neg_ll + self.reg_lambda * self._l2_penalty()
        return loss

    def predict(self, user_indices, item_indices=None, **kwargs):
        inputs = self._get_user_interactions(user_indices)
        logits = self._autoencoder(inputs)
        predictions = F.log_softmax(logits, dim=1)

        if item_indices is None:
            return predictions

        predictions = predictions.gather(1, item_indices.clamp(min=0))
        return predictions

    def _get_user_interactions(self, user_indices):
        rows = self._train_matrix[user_indices.cpu().numpy()].toarray().astype(np.float32)
        return torch.from_numpy(rows).to(self._device)

    def _l2_penalty(self):
        reg = torch.tensor(0.0, device=self._device)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                reg = reg + module.weight.pow(2).sum()
        return reg
