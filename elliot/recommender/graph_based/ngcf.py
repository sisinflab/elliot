"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from typing import Tuple
import torch
import torch_geometric
from torch import nn
from torch_sparse import SparseTensor

from elliot.dataset.samplers import PairWiseSampler
from elliot.recommender.base_recommender import GraphBasedRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.recommender.layers import SparseDropout, NGCFLayer


class NGCF(GraphBasedRecommender):
    """
    Neural Graph Collaborative Filtering

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3331184.3331267>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        weight_size: Tuple with number of units for each embedding propagation layer
        node_dropout: Tuple with dropout rate for each node
        message_dropout: Tuple with dropout rate for each embedding propagation layer

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NGCF:
          meta:
            save_recs: True
          learning_rate: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          n_layers: 1
          lambda_weights: 0.01
          weight_size: (64,)
          node_dropout: 0.0
          message_dropout: 0.5
    """

    # Model hyperparameters
    factors: int = 64
    n_layers: int = 1
    weight_size: Tuple[int, ...] = (64,)
    node_dropout: float = 0.0
    message_dropout: float = 0.5
    normalize: bool = True
    learning_rate: float = 0.0005
    lambda_weights: float = 0.01

    def __init__(self, data, params, seed, logger):
        super(NGCF, self).__init__(data, params, seed, logger)

        # Initialize the hidden dimensions
        self.weight_size_list = [self.factors] + list(self.weight_size)

        # Embeddings
        self.Gu = nn.Embedding(self._num_users, self.factors)
        self.Gi = nn.Embedding(self._num_items, self.factors)

        # Adjacency matrix
        self.adj = self.get_adj_mat()

        # Optionally define a dropout layer (optimized for sparse data)
        self.sparse_dropout = SparseDropout(self.node_dropout) if self.node_dropout > 0 else None

        # Initialize the propagation network
        propagation_network_list = []
        for i in range(self.n_layers):
            in_f = self.weight_size_list[i]
            out_f = self.weight_size_list[i + 1]
            propagation_network_list.append(
                (NGCFLayer(in_f, out_f, self.message_dropout), "x, edge_index -> x")
            )
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # Init embedding weights
        self.apply(xavier_normal_init)

        # Loss and optimizer
        self.log_sigmoid = nn.LogSigmoid()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Move to device
        self.to(self._device)

    def get_training_dataloader(self):
        dataloader = self._data.training_dataloader(PairWiseSampler, self._seed)
        return dataloader

    def forward(self):
        ego_embeddings = self.get_ego_embeddings(self.Gu, self.Gi)
        embeddings_list = [ego_embeddings]

        # Apply dropout if required from hyperparameters
        adj_matrix_current = self.adj
        if self.sparse_dropout is not None:
            adj_matrix_current = self.sparse_dropout(self.adj)

        # Forward each embedding through the sequential
        # propagation network
        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network.children():
            current_embeddings = layer_module(current_embeddings, adj_matrix_current)
            embeddings_list.append(current_embeddings)

        # Concatenate embeddings from all layers (including ego-embeddings)
        # along the feature dimension
        ngcf_all_embeddings = torch.stack(embeddings_list, dim=1)

        # Compute the mean across all stacked embeddings
        ngcf_all_embeddings = ngcf_all_embeddings.mean(dim=1, keepdim=False)

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self._num_users, self._num_items]
        )
        return user_all_embeddings, item_all_embeddings

    def train_step(self, batch, *args):
        user, pos, neg = [x.to(self._device) for x in batch]

        # Get propagated embeddings
        user_e_all, item_e_all = self.forward()

        # Get embeddings for current batch users and items
        u_embeddings = user_e_all[user]
        pos_embeddings = item_e_all[pos]
        neg_embeddings = item_e_all[neg]

        # Calculate BPR Loss
        xu_pos = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        xu_neg = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        reg = 0.5 * (self.Gu.weight[user].norm(2).pow(2) +
                     self.Gi.weight[pos].norm(2).pow(2) +
                     self.Gi.weight[neg].norm(2).pow(2)) / float(user.shape[0])

        loss = -torch.mean(self.log_sigmoid(xu_pos - xu_neg)) + self.lambda_weights * reg

        return loss

    def predict_full(self, user_indices):
        user_e_all, item_e_all = self.forward()

        # Select only the embeddings in the current batch
        u_embeddings_batch = user_e_all[user_indices]

        # Compute predictions
        predictions = torch.matmul(u_embeddings_batch, item_e_all.T)

        return predictions.to(self._device)

    def predict_sampled(self, user_indices, item_indices):
        user_e_all, item_e_all = self.forward()

        # Select only the embeddings in the current batch
        # and the candidate items
        u_embeddings_batch = user_e_all[user_indices]
        i_embeddings_candidate = item_e_all[item_indices.clamp(min=0)]

        # Compute predictions
        predictions = torch.einsum(
            "bi,bji->bj", u_embeddings_batch, i_embeddings_candidate
        )

        return predictions.to(self._device)

    def get_adj_mat(self):
        A = super().get_adj_mat()

        # Return the simple adjacency matrix
        # in case the 'normalize' flag is set to False
        if not self.normalize:
            return A

        # Symmetric Normalization: D^{-0.5} A D^{-0.5}
        # Convert to COO format for better work with rows, cols, and values
        A = A.to_torch_sparse_coo_tensor().coalesce()
        indices = A.indices()
        row, col = indices[0], indices[1]
        values = A.values()

        # Compute D^{-0.5} diagonal values
        deg = torch.zeros(A.size(0), dtype=values.dtype, device=values.device)
        deg.index_add_(0, row, values)
        # Add epsilon to avoid division by zero
        deg[deg == 0] = 1e-7
        deg_inv_sqrt = deg.pow(-0.5)

        # L = D^{-0.5} A D^{-0.5}
        new_val = deg_inv_sqrt[row] * values * deg_inv_sqrt[col]

        # Return the tensor as a SparseTensor
        normalized_adj = SparseTensor(
            row=row,
            col=col,
            value=new_val,
            sparse_sizes=A.shape
        )

        return normalized_adj
