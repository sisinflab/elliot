from typing import Any, Optional, Tuple
import torch
from torch import Tensor
from torch_geometric import EdgeIndex


class SparseAdjacency:
    """A sparse adjacency matrix backed by a `torch_geometric.EdgeIndex`.

    Exposes the subset of the `torch_sparse.SparseTensor` interface that the
    graph recommenders need (`matmul`, `coo`, `sum`, `set_value`, `to`),
    without depending on `torch_sparse` - a package only distributed through
    PyG's own wheel index - since `EdgeIndex` ships with `torch_geometric` on
    PyPI and is measurably faster on both CPU and GPU.

    Args:
        row (Tensor): Row indices of the non-zero entries.
        col (Tensor): Column indices of the non-zero entries.
        size (Tuple[int, int]): Shape of the matrix.
        value (Optional[Tensor]): Values of the non-zero entries. When omitted
            the matrix is unweighted and every entry is 1.
        is_sorted (bool): Whether the entries are already sorted by row and
            then by column. `EdgeIndex.matmul` requires that ordering, so
            leave this False unless the caller already sorted them.
    """

    def __init__(
        self,
        row: Tensor,
        col: Tensor,
        size: Tuple[int, int],
        value: Optional[Tensor] = None,
        is_sorted: bool = False,
    ):
        if value is None:
            value = torch.ones(row.numel(), dtype=torch.get_default_dtype(), device=row.device)

        if not is_sorted:
            order = torch.argsort(row * size[1] + col)
            row, col, value = row[order], col[order], value[order]

        self._row = row
        self._col = col
        self._value = value
        self._size = size
        self._edge_index = EdgeIndex(
            torch.stack([row, col]), sparse_size=size, sort_order="row"
        )

    def matmul(self, other: Tensor, reduce: str = "sum") -> Tensor:
        """Multiplies this matrix by a dense matrix.

        Args:
            other (Tensor): The dense right-hand side.
            reduce (str): The reduction to apply.

        Returns:
            Tensor: The product.
        """
        return self._edge_index.matmul(other, input_value=self._value, reduce=reduce)

    def coo(self) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the matrix in coordinate form.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Rows, columns and values.
        """
        return self._row, self._col, self._value

    def sum(self, dim: int = 1) -> Tensor:
        """Sums the values along one dimension, giving the weighted degree.

        Args:
            dim (int): 1 to sum along rows, 0 to sum along columns.

        Returns:
            Tensor: The per-node sum.
        """
        length = self._size[0] if dim == 1 else self._size[1]
        index = self._row if dim == 1 else self._col
        out = torch.zeros(length, device=self._value.device, dtype=self._value.dtype)
        return out.scatter_add_(0, index, self._value)

    def set_value(self, value: Tensor) -> "SparseAdjacency":
        """Returns a copy carrying different values for the same structure.

        Args:
            value (Tensor): The new values, in the order returned by `coo`.

        Returns:
            SparseAdjacency: The updated adjacency.
        """
        return SparseAdjacency(self._row, self._col, self._size, value, is_sorted=True)

    def to(self, device: Any) -> "SparseAdjacency":
        """Moves the adjacency to a device.

        Args:
            device (Any): The target device.

        Returns:
            SparseAdjacency: The adjacency on the requested device.
        """
        return SparseAdjacency(
            self._row.to(device),
            self._col.to(device),
            self._size,
            self._value.to(device),
            is_sorted=True,
        )
