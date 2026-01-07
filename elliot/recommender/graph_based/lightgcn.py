import torch
import torch_geometric
from torch import nn
from torch_geometric.nn import LGConv

from elliot.dataset.samplers import PairWiseSampler
from elliot.recommender.base_recommender import GraphBasedRecommender
from elliot.recommender.init import xavier_normal_init


class LightGCN(GraphBasedRecommender):
    """
    LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3397271.3401063>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        n_layers: Number of stacked propagation layers

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        LightGCN:
          meta:
            save_recs: True
          learning_rate: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          lambda_weights: 0.1
          n_layers: 2
    """

    # Model hyperparameters
    factors: int = 10
    n_layers: int = 1
    learning_rate: float = 0.0005
    lambda_weights: float = 0.01

    def __init__(self, data, params, seed, logger):
        super(LightGCN, self).__init__(data, params, seed, logger)

        # Embeddings
        self.Gu = nn.Embedding(self._num_users, self.factors, dtype=torch.float32)
        self.Gi = nn.Embedding(self._num_items, self.factors, dtype=torch.float32)

        # Adjacency matrix
        self.adj = self.get_adj_mat()

        # Propagation network
        propagation_network_list = []
        for _ in range(self.n_layers):
            propagation_network_list.append((LGConv(), "x, edge_index -> x"))
        self.propagation_network = torch_geometric.nn.Sequential(
            "x, edge_index", propagation_network_list
        )

        # Vectorized normalization for embedding
        self.alpha = torch.tensor([1 / (k + 1) for k in range(self.n_layers + 1)], device=self._device)

        # Loss and optimizer
        self.softplus = nn.functional.softplus
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Init embedding weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def get_training_dataloader(self):
        dataloader = self._data.training_dataloader(PairWiseSampler, self._seed)
        return dataloader

    def forward(self):
        ego_embeddings = self.get_ego_embeddings(self.Gu, self.Gi)
        embeddings_list = [ego_embeddings]

        # This will handle the propagation layer by layer.
        # This is used later to correctly multiply each layer by
        # the corresponding value of alpha
        current_embeddings = ego_embeddings
        for layer_module in self.propagation_network.children():
            current_embeddings = layer_module(current_embeddings, self.adj)
            embeddings_list.append(current_embeddings)

        # Aggregate embeddings using the alpha value
        lightgcn_all_embeddings = torch.zeros_like(ego_embeddings, device=self._device)
        for k in range(len(embeddings_list)):
            lightgcn_all_embeddings += embeddings_list[k] * self.alpha[k]

        # Split into user and item embeddings
        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self._num_users, self._num_items]
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
                     self.Gi.weight[neg].norm(2).pow(2)) / float(batch[0].shape[0])

        loss = torch.mean(self.softplus(xu_neg - xu_pos)) + self.lambda_weights * reg

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
