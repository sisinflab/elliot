"""Payload -> tensor/matrix: turns an already-built `EmbeddingPayload`/`TextPayload`/
`GraphPayload` into the dense array/sparse matrix/tensor shape a recommender wants to
compute with. None of these touch the filesystem or build a payload themselves - see
`.build` for that half of the pipeline.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import torch

from elliot.dataset.modular_loaders.formats import EmbeddingPayload, GraphPayload, TextPayload


def embedding_to_dense(payload: EmbeddingPayload, ids: Optional[Iterable[Any]] = None) -> np.ndarray:
    """Materialize an `EmbeddingPayload` as a dense numpy matrix.

    If `ids` is given, only those rows are returned, in the given order (looked up via
    `payload.id_map`); otherwise every row is returned in on-disk/matrix order.

    Args:
        payload (EmbeddingPayload): The payload to materialize.
        ids (Iterable[Any], optional): Domain ids selecting (and ordering) the rows
            to return. Defaults to None, returning every row in on-disk/matrix order.

    Returns:
        np.ndarray: The dense matrix.

    Raises:
        ValueError: If `payload` has no `dense`/`sparse`/`row_loader` data to materialize.
    """
    # row_loader-backed: read (and stack) each requested row on demand
    if payload.row_loader is not None:
        row_ids = list(ids) if ids is not None else list(payload.row_ids or [])
        rows = [
            payload.row_loader(payload.id_map[i] if payload.id_map else i)
            for i in row_ids
        ]
        return np.stack(rows, axis=0) if rows else np.empty((0, 0))

    # Already dense, or densify a sparse payload
    if payload.dense is not None:
        dense = payload.dense
    elif payload.sparse is not None:
        dense = np.asarray(payload.sparse.todense())
    else:
        raise ValueError("EmbeddingPayload has no dense/sparse/row_loader data to materialize.")

    # No id filtering requested: return every row as-is
    if ids is None or payload.id_map is None:
        return dense
    idx = [payload.id_map[i] for i in ids]
    return dense[idx]


def embedding_to_sparse(payload: EmbeddingPayload) -> sp.csr_matrix:
    """Materialize an `EmbeddingPayload` as a `scipy.sparse.csr_matrix`.

    Args:
        payload (EmbeddingPayload): The payload to materialize.

    Returns:
        sp.csr_matrix: The sparse matrix.

    Raises:
        ValueError: If `payload` has no `dense`/`sparse` data to convert (i.e. it is
            `row_loader`-backed).
    """
    if payload.sparse is not None:
        return payload.sparse.tocsr()
    if payload.dense is not None:
        return sp.csr_matrix(payload.dense)
    raise ValueError(
        "EmbeddingPayload has no dense/sparse data to convert (it's row_loader-backed; "
        "use embedding_to_dense(payload, ids=...) instead)."
    )


def embedding_to_tensor(
    payload: EmbeddingPayload,
    ids: Optional[Iterable[Any]] = None,
    device: Optional[Any] = None,
) -> torch.Tensor:
    """Materialize an `EmbeddingPayload` as a dense `torch.Tensor`.

    Args:
        payload (EmbeddingPayload): The payload to materialize.
        ids (Iterable[Any], optional): Domain ids selecting (and ordering) the rows
            to return. Defaults to None, returning every row in on-disk/matrix order.
        device (Any, optional): Device to move the resulting tensor to. Defaults to
            None, keeping the tensor on its default device.

    Returns:
        torch.Tensor: The dense tensor.
    """
    dense = embedding_to_dense(payload, ids=ids)
    tensor = torch.as_tensor(dense, dtype=torch.float32)
    return tensor.to(device) if device is not None else tensor


def feature_map_to_sparse(
    feature_map: Dict[Any, List[Any]],
    id_map: Dict[Any, int],
    n_cols: int,
) -> sp.csr_matrix:
    """Build a one-hot `(len(id_map), n_cols)` sparse matrix from a categorical
    multi-hot feature map (`dict[id -> list[feature id]]`) plus a row-id index - the
    pattern duplicated today across `AttributeItemKNN`/`AttributeUserKNN`/`VSM`/`FM`.

    Args:
        feature_map (Dict[Any, List[Any]]): Mapping from an id to its list of
            (already contiguous) feature ids.
        id_map (Dict[Any, int]): Mapping from a domain id to its row index.
        n_cols (int): Number of columns (distinct feature ids) in the matrix.

    Returns:
        sp.csr_matrix: The one-hot sparse matrix.
    """
    rows, cols = [], []
    for entity_id, feature_ids in feature_map.items():
        # Skip ids outside id_map (e.g. filtered out of the domain)
        row = id_map.get(entity_id)
        if row is None:
            continue
        for f in feature_ids:
            rows.append(row)
            cols.append(f)

    # One-hot: every (row, feature) pair gets weight 1.0
    data = np.ones(len(rows), dtype=np.float32)
    return sp.csr_matrix((data, (rows, cols)), shape=(len(id_map), n_cols))


def text_to_padded_ids(payload: TextPayload, max_len: int, pad_value: int = 0):
    """Pad/truncate `payload.tokens` (`dict[id -> list[int]]`) to a fixed-length
    `(n_ids, max_len)` int array, plus the (pre-padding) sequence lengths, both ordered
    according to `payload.id_map` (falling back to `payload.tokens`' own order).

    Args:
        payload (TextPayload): The payload whose tokenized sequences to pad/truncate.
        max_len (int): Fixed sequence length to pad/truncate every row to.
        pad_value (int): Value used to fill padding positions. Defaults to 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The `(n_ids, max_len)` padded token array and
            the (pre-padding) sequence lengths.
    """
    ids = (
        list(payload.id_map.keys())
        if payload.id_map else list((payload.tokens or {}).keys())
    )
    padded = np.full((len(ids), max_len), pad_value, dtype=np.int64)
    lengths = np.zeros(len(ids), dtype=np.int64)

    # Truncate each id's token sequence to max_len and copy it into its padded row
    for row, entity_id in enumerate(ids):
        seq = (payload.tokens or {}).get(entity_id, [])
        length = min(len(seq), max_len)
        padded[row, :length] = seq[:length]
        lengths[row] = length

    return padded, lengths


def graph_triples_to_edge_index(payload: GraphPayload) -> torch.Tensor:
    """Return a `(2, n_edges)` `torch.LongTensor` edge index from a `GraphPayload`,
    treating each (head, tail) pair as an edge (relations are dropped here - use
    `graph_triples_to_adjacency` per-relation, or `payload.relations` directly, for
    relation-aware propagation).

    Args:
        payload (GraphPayload): The payload whose (head, tail) pairs to vectorize.

    Returns:
        torch.Tensor: The `(2, n_edges)` edge index.
    """
    heads = torch.as_tensor(payload.heads, dtype=torch.long)
    tails = torch.as_tensor(payload.tails, dtype=torch.long)
    return torch.stack([heads, tails], dim=0)


def graph_item_entity_ids_to_tensor(payload: GraphPayload, device: Optional[Any] = None) -> torch.Tensor:
    """Materialize a `GraphPayload.item_entity_map` into a dense `int64`
    `torch.LongTensor` indexed by private item id `0..n_items-1` - the bridge a
    recommender needs to gather its own (private-index-ordered) item embeddings
    from the KG's shared entity embedding table.

    `payload` is expected to already be this fold's private-id view (as returned by
    `Interactions.get_side_info()`, via `remap_graph_payload` - see
    `elliot.dataset.modular_loaders.remap`), i.e. `item_entity_map` keyed by private
    item id, not the loader's raw public-id-keyed one.

    Args:
        payload (GraphPayload): The (already fold-private) payload whose
            `item_entity_map` to materialize.
        device (Any, optional): Device to move the resulting tensor to. Defaults to
            None, keeping the tensor on its default device.

    Returns:
        torch.Tensor: The `int64` entity id tensor, indexed by private item id.
    """
    n_items = len(payload.item_entity_map)
    tensor = torch.tensor(
        [payload.item_entity_map[i] for i in range(n_items)], dtype=torch.long
    )
    return tensor.to(device) if device is not None else tensor


def dedupe_edges(
    edge_index: torch.Tensor,
    edge_attr: Optional[torch.Tensor] = None,
    device: Optional[Any] = None,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """Remove duplicate rows from a `(2, n_edges)` edge index (and its optional
    parallel per-edge attribute, e.g. a relation type) - payload-agnostic building
    block for any model that must not double-count a repeated edge. Combine with
    `graph_triples_to_edge_index`'s output and `payload.relations` for a
    relation-aware KG-propagation model (e.g. KGIN), which expects one edge per
    distinct (head, relation, tail) triple, not per occurrence in the raw data.

    Args:
        edge_index (torch.Tensor): The `(2, n_edges)` edge index to de-duplicate.
        edge_attr (torch.Tensor, optional): A parallel `(n_edges,)` per-edge
            attribute (e.g. relation type); when given, two edges are only
            considered duplicates if they also share the same attribute value.
            Defaults to None.
        device (Any, optional): Device to move the result(s) to. Defaults to None,
            keeping them on their default device.

    Returns:
        torch.Tensor | Tuple[torch.Tensor, torch.Tensor]: The de-duplicated
            `(2, n_unique_edges)` edge index alone, or paired with the matching
            `(n_unique_edges,)` attribute tensor when `edge_attr` was given.
    """
    columns = [edge_index[0], edge_index[1]] + ([edge_attr] if edge_attr is not None else [])
    unique_rows = torch.unique(torch.stack(columns, dim=1), dim=0)

    deduped_index = unique_rows[:, :2].T.contiguous()
    if device is not None:
        deduped_index = deduped_index.to(device)

    if edge_attr is None:
        return deduped_index

    deduped_attr = unique_rows[:, 2].contiguous()
    if device is not None:
        deduped_attr = deduped_attr.to(device)
    return deduped_index, deduped_attr


def graph_triples_to_adjacency(payload: GraphPayload, n_nodes: int) -> sp.csr_matrix:
    """Build a `(n_nodes, n_nodes)` sparse adjacency matrix from a `GraphPayload`
    (unweighted, one entry per (head, tail) edge, symmetrized).

    Args:
        payload (GraphPayload): The payload whose (head, tail) pairs to vectorize.
        n_nodes (int): Total number of nodes, i.e. the matrix's side length.

    Returns:
        sp.csr_matrix: The `(n_nodes, n_nodes)` sparse adjacency matrix.
    """
    heads, tails = payload.heads, payload.tails

    # Symmetrize: add both (head, tail) and (tail, head) edges
    rows = (
        np.concatenate([heads, tails])
        if len(heads) else np.empty(0, dtype=np.int64)
    )
    cols = (
        np.concatenate([tails, heads])
        if len(tails) else np.empty(0, dtype=np.int64)
    )
    data = np.ones(len(rows), dtype=np.float32)

    return sp.csr_matrix((data, (rows, cols)), shape=(n_nodes, n_nodes))
