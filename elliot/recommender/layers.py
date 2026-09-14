from typing import List, Tuple, Union
import torch
from torch import nn, Tensor
from torch.nn.init import zeros_, xavier_normal_

from elliot.recommender.init import normal_init
from elliot.recommender.modules import SparseAdjacency


def get_activation(activation: str = "relu") -> nn.Module:
    """Get the activation function using enum.

    Args:
        activation (str): The activation layer to retrieve.

    Returns:
        Module: The activation layer requested.

    Raises:
        ValueError: If the activation is not known or supported.
    """
    match activation:
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "relu":
            return nn.ReLU()
        case "leakyrelu":
            return nn.LeakyReLU()
        case _:
            raise ValueError("Activation function not supported.")


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        noise = torch.randn_like(x) * self.stddev
        return x + noise


class MLP(nn.Module):
    """Simple implementation of MultiLayer Perceptron.

    Args:
        layers (List[int]): The hidden layers size list.
        dropout (float): The dropout probability.
        activation (str): The activation function to apply.
        batch_normalization (bool): Wether or not to apply batch normalization.
        initialize (bool): Wether or not to initialize the weights.
        last_activation (bool): Wether or not to keep last non-linearity function.
    """

    def __init__(
        self,
        layers: List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_normalization: bool = False,
        initialize: bool = False,
        last_activation: bool = True,
    ):
        super(MLP, self).__init__()
        mlp_modules: List[nn.Module] = []
        for input_size, output_size in zip(layers[:-1], layers[1:]):
            mlp_modules.append(nn.Dropout(p=dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if batch_normalization:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            if activation:
                mlp_modules.append(get_activation(activation))
        if activation is not None and not last_activation:
            mlp_modules.pop()
        self.mlp_layers = nn.Sequential(*mlp_modules)
        if initialize:
            self.apply(normal_init)

    def forward(self, input_feature: torch.Tensor):
        """Simple forwarding, input tensor will pass
        through all the MLP layers.
        """
        return self.mlp_layers(input_feature)


class SparseDropout(nn.Module):
    """Dropout layer for sparse tensors.

    Args:
        p (float): Dropout rate. Values accepted in range [0, 1].

    Raises:
        ValueError: If p is not in range.
    """

    def __init__(self, p: float):
        super(SparseDropout, self).__init__()
        if not (0 <= p <= 1):
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p

    def forward(self, X: Union[Tensor, SparseAdjacency]) -> Union[Tensor, SparseAdjacency]:
        """Apply dropout to a sparse matrix.

        Args:
            X (Union[Tensor, SparseAdjacency]): The input matrix - either a
                `SparseAdjacency` or a plain `torch.sparse_coo_tensor`.

        Returns:
            Union[Tensor, SparseAdjacency]: The matrix after the dropout,
                rescaled so that the expected value is unchanged, in the same
                representation as the input.
        """
        if self.p == 0 or not self.training:
            return X

        if isinstance(X, SparseAdjacency):
            _, _, values = X.coo()
            keep = (torch.rand(values.numel(), device=values.device) > self.p).to(values.dtype)
            return X.set_value(values * keep / (1 - self.p))

        # Plain torch sparse COO tensor
        X = X.coalesce()
        indices = X.indices()
        values = X.values()

        random_tensor = torch.rand(values.numel(), device=X.device)
        dropout_mask = random_tensor > self.p

        out_indices = indices[:, dropout_mask]
        out_values = values[dropout_mask] / (1 - self.p)

        return torch.sparse_coo_tensor(
            out_indices, out_values, X.shape, device=X.device
        ).coalesce()


class EdgeDropout(nn.Module):
    """Dropout layer for graph edges: prunes a `p` fraction of a `(2, n_edges)`
    edge index, carrying along the parallel per-edge relation-type tensor so
    the two stay in sync.

    Args:
        p (float): Dropout rate. Values accepted in range [0, 1].

    Raises:
        ValueError: If p is not in range.
    """

    def __init__(self, p: float):
        super(EdgeDropout, self).__init__()
        if not (0 <= p <= 1):
            raise ValueError(
                f"Dropout probability has to be between 0 and 1, but got {p}"
            )
        self.p = p

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply dropout to an edge index and its parallel relation-type tensor.

        Args:
            edge_index (Tensor): The `(2, n_edges)` `(head, tail)` edge index.
            edge_type (Tensor): The parallel `(n_edges,)` relation-type tensor.

        Returns:
            Tuple[Tensor, Tensor]: The pruned `(edge_index, edge_type)` pair.
        """
        if self.p == 0 or not self.training:
            return edge_index, edge_type

        n_edges = edge_index.shape[1]
        keep = torch.randperm(n_edges, device=edge_index.device)[:int(n_edges * (1 - self.p))]
        return edge_index[:, keep], edge_type[keep]


class RelationAwareEdgeDropout(nn.Module):
    """Edge dropout applied independently within every relation type, so a rare
    relation isn't starved by a single global draw over the whole edge set - a
    per-relation wrapper around a single shared `EdgeDropout`.

    Args:
        p (float): Dropout rate, forwarded to the underlying `EdgeDropout`.
    """

    def __init__(self, p: float):
        super().__init__()
        self.edge_dropout = EdgeDropout(p)

    def forward(self, edge_index: Tensor, edge_type: Tensor) -> Tuple[Tensor, Tensor]:
        """Apply per-relation dropout to an edge index and its parallel
        relation-type tensor.

        Args:
            edge_index (Tensor): The `(2, n_edges)` `(head, tail)` edge index.
            edge_type (Tensor): The parallel `(n_edges,)` relation-type tensor.

        Returns:
            Tuple[Tensor, Tensor]: The pruned `(edge_index, edge_type)` pair.
        """
        sampled_index, sampled_type = [], []
        for rel in torch.unique(edge_type):
            mask = edge_type == rel
            idx, typ = self.edge_dropout(edge_index[:, mask], edge_type[mask])
            sampled_index.append(idx)
            sampled_type.append(typ)
        return torch.cat(sampled_index, dim=1), torch.cat(sampled_type, dim=0)


class RelationWeightedMeanAggregator(nn.Module):
    """Relation-weighted mean aggregation over a knowledge graph: for every
    entity, the mean over its incoming `(head, relation, tail)` edges of
    `entity_emb[tail] * relation_weight[relation]`. Relation id `0` is assumed
    reserved for a relation with no learned embedding (e.g. a non-KG
    "interacts" relation) and is excluded from `relation_weight` indexing.
    """

    def forward(
        self, entity_emb: Tensor, edge_index: Tensor, edge_type: Tensor, relation_weight: Tensor
    ) -> Tensor:
        """
        Args:
            entity_emb (Tensor): The `(n_entities, channel)` entity embeddings.
            edge_index (Tensor): The `(2, n_edges)` `(head, tail)` edge index.
            edge_type (Tensor): The parallel `(n_edges,)` relation-type tensor.
            relation_weight (Tensor): The `(n_relations - 1, channel)` per-relation weights.

        Returns:
            Tensor: The `(n_entities, channel)` aggregated entity embeddings.
        """
        n_entities, channel = entity_emb.shape
        head, tail = edge_index[0], edge_index[1]
        edge_relation_emb = relation_weight[edge_type - 1]
        neigh_relation_emb = entity_emb[tail] * edge_relation_emb

        entity_agg = torch.zeros(n_entities, channel, device=entity_emb.device)
        entity_agg.index_add_(0, head, neigh_relation_emb)
        neigh_count = torch.zeros(n_entities, device=entity_emb.device)
        neigh_count.index_add_(0, head, torch.ones(head.shape[0], device=entity_emb.device))
        return entity_agg / neigh_count.clamp(min=1.0).unsqueeze(-1)


class NGCFLayer(nn.Module):
    """Implementation of a single layer of NGCF propagation.
    - First term: GCN-like aggregation of neighbors.
    - Second term: Element-wise product capturing interaction between ego-embedding and aggregated neighbors.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        message_dropout (float): The dropout value.
    """

    def __init__(
        self, in_features: int, out_features: int, message_dropout: float = 0.0
    ):
        super(NGCFLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Weight matrices for the two terms
        self.W1 = nn.Parameter(torch.Tensor(in_features, out_features))
        self.W2 = nn.Parameter(torch.Tensor(in_features, out_features))

        # Biases for the two terms
        self.b1 = nn.Parameter(torch.Tensor(1, out_features))
        self.b2 = nn.Parameter(torch.Tensor(1, out_features))

        # LeakyReLU non-linearity and dropout layer
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.dropout = nn.Dropout(p=message_dropout)

        self.init_parameters()

    def init_parameters(self):
        xavier_normal_(self.W1.data)
        xavier_normal_(self.W2.data)
        zeros_(self.b1.data)
        zeros_(self.b2.data)

    def forward(self, ego_embeddings: Tensor, adj_matrix: SparseAdjacency) -> Tensor:
        """
        Performs a single NGCF propagation step.

        Args:
            ego_embeddings (Tensor): Current embeddings of all nodes (users + items).
            adj_matrix (SparseAdjacency): Normalized adjacency matrix (A_hat).

        Returns:
            Tensor: Propagated embeddings for the next layer.
        """
        laplacian_embeddings = adj_matrix.matmul(ego_embeddings)

        # First term: (A_hat + I) * E * W1 + b1
        first_term = torch.matmul(ego_embeddings + laplacian_embeddings, self.W1) + self.b1

        # Second term: (A_hat * E) element-wise product E * W2 + b2
        second_term = torch.mul(ego_embeddings, laplacian_embeddings)
        second_term = torch.matmul(second_term, self.W2) + self.b2

        # Combine terms, apply activation, dropout, and normalize
        output_embeddings = self.leaky_relu(first_term + second_term)
        output_embeddings = self.dropout(output_embeddings)
        output_embeddings = nn.functional.normalize(output_embeddings, p=2, dim=1)

        return output_embeddings
