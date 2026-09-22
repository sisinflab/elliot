"""
Module description:

"""


from typing import Tuple
import torch
import torch_geometric
from torch import nn

from elliot.dataset import Interactions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import GraphBasedRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.recommender.layers import SparseDropout, NGCFLayer
from elliot.recommender.losses import BPRLoss, EmbLoss
from elliot.utils.registry import model_registry


@model_registry.register()
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
    # weight_size: int = 64
    node_dropout: float = 0.0
    message_dropout: float = 0.5
    normalize: bool = True
    learning_rate: float = 0.0005
    lambda_weights: float = 0.01

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super(NGCF, self).__init__(params, seed, interactions, *args, **kwargs)

        # Initialize the hidden dimensions
        self.weight_size_list = [self.factors] * (self.n_layers + 1)

        # Embeddings
        self.Gu = nn.Embedding(self._num_users, self.factors)
        self.Gi = nn.Embedding(self._num_items, self.factors)

        # Adjacency matrix
        self.adj = self.get_adj_mat(normalize=self.normalize)

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

        # Loss and optimizer
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Sampler configuration
        self.sampler_config = {
            "name": "PairWiseSampler"
        }

        # Init embedding weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def forward(self):
        ego_embeddings = self.get_ego_embeddings(self.Gu, self.Gi)
        embeddings_list = [ego_embeddings]

        # Apply dropout if required from hyperparameters
        adj_matrix_current = self.adj
        if self.sparse_dropout is not None:
            adj_matrix_current = self.sparse_dropout(self.adj)

        # Forward each embedding through the sequential propagation network
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

        main_loss = self.bpr_loss(xu_pos, xu_neg)
        reg_loss = self.lambda_weights * self.reg_loss(
            self.Gu.weight[user], self.Gi.weight[pos], self.Gi.weight[neg]
        )

        return main_loss + reg_loss

    def predict(self, user_indices, item_indices=None, **kwargs):
        user_e_all, item_e_all = self.propagate_embeddings()

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
        return predictions
