"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from typing import Tuple
import torch
from torch import nn

from elliot.dataset.samplers import MFPointWisePosNegSampler
from elliot.recommender.init import normal_init
from elliot.recommender.base_recommender import GeneralRecommender
from elliot.recommender.layers import MLP


class NeuMF(GeneralRecommender):
    """
    Neural Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/1708.05031>`_

    Args:
        mf_factors: Number of MF latent factors
        mlp_factors: Number of MLP latent factors
        mlp_hidden_size: List of units for each layer
        lr: Learning rate
        dropout: Dropout rate
        is_mf_train: Whether to train the MF embeddings
        is_mlp_train: Whether to train the MLP layers

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NeuMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          mf_factors: 10
          mlp_factors: 10
          mlp_hidden_size: (64,32)
          dropout: 0.0
          is_mf_train: True
          is_mlp_train: True
          learning_rate: 1e-5
          lambda_weights: 0.1
    """

    # Model hyperparameters
    embed_mf_size: int = 10
    embed_mlp_size: int = 10
    mlp_hidden_size: Tuple[int, ...] = (64, 32)
    dropout: float = 0.0
    is_mf_train: bool = True
    is_mlp_train: bool = True
    learning_rate: float = 1e-5
    lambda_weights: float = 0.1
    m: int = 0
    batch_eval_items: int = 256

    def __init__(self, data, params, seed, logger):
        super(NeuMF, self).__init__(data, params, seed, logger)

        # MF embeddings
        self.user_mf_embedding = nn.Embedding(self._num_users, self.embed_mf_size, dtype=torch.float32)
        self.item_mf_embedding = nn.Embedding(self._num_items, self.embed_mf_size, dtype=torch.float32)

        # MLP embeddings
        self.user_mlp_embedding = nn.Embedding(self._num_users, self.embed_mlp_size, dtype=torch.float32)
        self.item_mlp_embedding = nn.Embedding(self._num_items, self.embed_mlp_size, dtype=torch.float32)

        # MLP layers
        self.mlp_layers = MLP(
            [2 * self.embed_mlp_size] + list(self.mlp_hidden_size), self.dropout
        )

        # Final prediction layer
        if self.is_mf_train and self.is_mlp_train:
            self.predict_layer = nn.Linear(self.embed_mf_size + self.mlp_hidden_size[-1], 1)
        elif self.is_mf_train:
            self.predict_layer = nn.Linear(self.embed_mf_size, 1)
        else:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)

        # Activation function, loss and optimizer
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self.apply(normal_init, mean=0.0, std=0.01)

        # Move to device
        self.to(self._device)

    def get_training_dataloader(self):
        dataloader = self._data.training_dataloader(MFPointWisePosNegSampler, self._seed, m=self.m)
        return dataloader

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)

        if self.is_mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)

        if self.is_mlp_train:
            mlp_input = torch.cat((user_mlp_e, item_mlp_e), -1)
            mlp_output = self.mlp_layers(mlp_input)

        if self.is_mf_train and self.is_mlp_train:
            combined = torch.cat((mf_output, mlp_output), -1)
            output = self.predict_layer(combined)
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        else:
            output = self.predict_layer(mlp_output)

        return output.squeeze(-1)

    def train_step(self, batch, *args):
        user, pos, label = [x.to(self._device) for x in batch]

        output = self.forward(user, pos)
        loss = self.loss(label.float(), output)

        return loss

    def predict_full(self, user_indices):
        batch_size = len(user_indices)
        preds = []

        for s in range(0, self._num_items, self.batch_eval_items):
            e = min(s + self.batch_eval_items, self._num_items)
            items_block = torch.arange(s, e)

            # Expand user_indices and items_block to create all user-item pairs
            # within this block.
            users_block = (
                user_indices.unsqueeze(1).expand(-1, e - s).reshape(-1)
            )
            items_block_expanded = (
                items_block.unsqueeze(0).expand(batch_size, -1).reshape(-1)
            )

            preds_block = self.sigmoid(self.forward(users_block, items_block_expanded))
            preds.append(preds_block.view(batch_size, e - s))

        predictions = torch.cat(preds, dim=1)

        return predictions.to(self._device)

    def predict_sampled(self, user_indices, item_indices):
        batch_size, pad_seq = item_indices.size()

        # Prepare user and item indices for forward pass
        users_expanded = user_indices.unsqueeze(1).expand(-1, pad_seq).reshape(-1)
        items_expanded = item_indices.clamp(min=0).reshape(-1)

        # Compute predictions using the forward pass
        predictions_flat = self.forward(users_expanded, items_expanded)
        predictions = predictions_flat.view(batch_size, pad_seq)

        # Apply sigmoid to get scores between 0 and 1
        predictions = self.sigmoid(predictions)

        return predictions.to(self._device)
