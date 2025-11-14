import typing as t
import torch
from torch import nn, Tensor
from torch.nn.init import zeros_, xavier_normal_
from torch_sparse import SparseTensor

from elliot.recommender.init import normal_init


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
        layers: t.List[int],
        dropout: float = 0.0,
        activation: str = "relu",
        batch_normalization: bool = False,
        initialize: bool = False,
        last_activation: bool = True,
    ):
        super(MLP, self).__init__()
        mlp_modules: t.List[nn.Module] = []
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

    def forward(self, X: SparseTensor) -> SparseTensor:
        """Apply dropout to SparseTensor.

        Args:
            X (SparseTensor): The input tensor.

        Returns:
            SparseTensor: The tensor after the dropout.
        """
        if self.p == 0 or not self.training:
            return X

        # Get indices and values of the sparse tensor
        X = X.to_torch_sparse_coo_tensor().coalesce()
        indices = X.indices()
        values = X.values()

        # Calculate number of non-zero elements
        n_nonzero_elems = len(values)

        # Create a dropout mask
        random_tensor = torch.rand(n_nonzero_elems, device=X.device)
        dropout_mask = (random_tensor > self.p)

        # Apply mask and scale
        out_indices = indices[:, dropout_mask]
        out_values = values[dropout_mask] / (1 - self.p)

        # Return the tensor as a SparseTensor
        filtered_adj = SparseTensor(
            row=out_indices[0],
            col=out_indices[1],
            value=out_values,
            sparse_sizes=X.shape
        )

        return filtered_adj


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

    def forward(self, ego_embeddings: Tensor, adj_matrix: SparseTensor) -> Tensor:
        """
        Performs a single NGCF propagation step.

        Args:
            ego_embeddings (Tensor): Current embeddings of all nodes (users + items).
            adj_matrix (SparseTensor): Normalized adjacency matrix (A_hat).

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
