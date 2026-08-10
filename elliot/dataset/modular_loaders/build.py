"""Raw data -> canonical payload: turns already-*read* raw data (a folder listing, a
parsed JSON dict, a list of KG triples, ...) into one of the canonical payloads (see
`elliot.dataset.modular_loaders.formats`). Every function here is a pure, in-memory
transform; reading the raw file/folder itself is `elliot.utils.read.Reader`'s job.
"""

import os
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np
import scipy.sparse as sp

from elliot.dataset.modular_loaders.materialize import feature_map_to_sparse
from elliot.dataset.modular_loaders.formats import EmbeddingPayload, GraphPayload
from elliot.utils.enums import Materialization
from elliot.utils.read import Reader


def public_id_map(ids: Iterable[Any]) -> Dict[Any, int]:
    """Assign a deterministic `0..n-1` row index to every id in `ids`, in sorted order.

    This is the "public" numbering every payload-building function below keys its
    `id_map` by (see the module docstring in `elliot.dataset.modular_loaders.adapters`
    for what "public" means here): every loader that turns a raw, already-filtered id
    domain into a fresh row index -- rather than reusing an existing `id_map` it already
    has, like `rows_to_embedding_payload`'s callers do -- should call this instead of
    hand-rolling `{i: idx for idx, i in enumerate(sorted(ids))}` itself, so every loader
    agrees on the same (sorted, hence reproducible run-to-run) row order.
    """
    return {entity_id: idx for idx, entity_id in enumerate(sorted(ids))}


def rows_to_embedding_payload(
    materialization: Materialization,
    id_map: Dict[Any, int],
    shape: Optional[Tuple[int, ...]],
    row_reader: Callable[[Any, Optional[str]], np.ndarray],
) -> EmbeddingPayload:
    """Generic `Materialization` dispatcher for any "one row per id, read on demand"
    source: `row_reader(entity_id, mmap_mode)` must return the raw row for `entity_id`,
    honoring `mmap_mode` (`None` for a plain read, `"r"` for a memory-mapped one) when
    the underlying source supports it (e.g. `Reader.read_npy`).

    `MEMORY` eagerly calls `row_reader` for every id and returns one fully materialized
    dense matrix. `LAZY`/`MMAP` instead return a `row_loader` that defers each call
    until a row is actually accessed -- `LAZY` with `mmap_mode=None` (a plain read,
    copied fresh every access, no lingering file handle), `MMAP` with `mmap_mode="r"`
    (memory-mapped, read-only, backed by the OS page cache across repeated accesses to
    the same id).

    This is the one place the actual MEMORY/LAZY/MMAP branching lives; every loader
    that reads one row per id -- whatever the underlying source -- calls this instead
    of reimplementing the dispatch (see `npy_folder_to_embedding_payload` below, and
    `InteractionsTextualAttributes.load()`, whose row identity and on-disk filename use
    two different id schemes so it can't reuse `npy_folder_to_embedding_payload`
    itself, but still shares this dispatch).
    """
    row_ids = sorted(id_map, key=id_map.get)
    full_shape = (len(id_map),) + tuple(shape) if shape else (len(id_map),)

    if materialization == Materialization.MEMORY:
        dense = np.empty(full_shape)
        for entity_id, row in id_map.items():
            dense[row] = row_reader(entity_id, None)
        return EmbeddingPayload(
            dense=dense, row_ids=row_ids, id_map=dict(id_map), shape=dense.shape
        )

    mmap_mode = "r" if materialization == Materialization.MMAP else None
    inverse = {row: entity_id for entity_id, row in id_map.items()}

    def row_loader(
        row_idx,
        _inverse=inverse,
        _reader=row_reader,
        _mmap_mode=mmap_mode
    ):
        return _reader(_inverse[row_idx], _mmap_mode)

    return EmbeddingPayload(
        row_loader=row_loader,
        row_ids=row_ids,
        id_map=dict(id_map),
        shape=full_shape
    )


def npy_folder_to_embedding_payload(
    folder_path: str,
    id_map: Dict[int, int],
    shape: Optional[Tuple[int, ...]],
    materialization: Materialization = Materialization.MMAP,
    reader: Optional[Reader] = None,
) -> EmbeddingPayload:
    """Build an `EmbeddingPayload` from a folder holding one `.npy` file per id (see
    `Reader.discover_npy_ids`, which a loader's `discover()` uses to build `id_map`/
    `shape` ahead of this call). A thin `row_reader` -- "read `{folder_path}/{id}.npy`"
    -- over the shared `rows_to_embedding_payload` dispatch; see that function for how
    `materialization` is actually handled. Every actual file read goes through `reader`
    (a fresh, default `Reader` if the caller doesn't already have one).
    """
    reader = reader or Reader()

    def row_reader(entity_id, mmap_mode, _folder=folder_path, _reader=reader):
        return _reader.read_npy(os.path.join(_folder, f"{entity_id}.npy"), mmap_mode=mmap_mode)

    return rows_to_embedding_payload(materialization, id_map, shape, row_reader)


def raw_feature_map_to_embedding_payload(feature_map: Dict[Any, List[Any]], items: Iterable[Any]) -> EmbeddingPayload:
    """Build a categorical multi-hot `EmbeddingPayload` from a raw `dict[id -> list[feature
    id]]` whose feature ids are not yet a contiguous `0..n` column index (this function
    assigns that index itself) -- the shape produced by `ItemAttributes`, `ChainedKG`,
    `KAHFMLoader` and `KGFlexLoader` alike, whichever raw KG/attribute format each reads.
    """
    id_map = public_id_map(items)
    row_ids = sorted(id_map, key=id_map.get)
    features = sorted({f for i in row_ids for f in feature_map.get(i, [])})
    public_features = {f: idx for idx, f in enumerate(features)}

    translated_map = {i: [public_features[f] for f in feature_map.get(i, [])] for i in row_ids}
    sparse = feature_map_to_sparse(translated_map, id_map, len(features))

    return EmbeddingPayload(
        sparse=sparse,
        row_ids=row_ids,
        id_map=id_map,
        col_ids=features,
        shape=sparse.shape
    )


def _resolve_pairwise_id(key: Any) -> Any:
    """Coerce a raw JSON key/value into the same id type `Reader.read_mapping`-style
    id sets use: `int` when possible (even via a `"1.0"`-style float string), else the
    original string. Shared by `pairwise_raw_to_embedding_payload` and
    `pairwise_ids_from_raw` so both agree on what an id looks like.
    """
    s = str(key)
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return s


def pairwise_ids_from_raw(raw: Dict[str, Any]) -> Set[Any]:
    """Collect the full id domain referenced by a pairwise JSON dict (item-item/
    user-user similarity or sentiment, already parsed via `Reader.read_json`), in
    either of the two raw layouts `pairwise_raw_to_embedding_payload` understands,
    *without* building the sparse matrix -- just enough to let a loader's `discover()`
    step know its own users/items domain ahead of the (potentially much larger)
    `load()` pass.
    """
    ids: Set[Any] = set()
    for key, value in raw.items():
        if isinstance(value, list):
            ids.add(_resolve_pairwise_id(key))
            ids.update(_resolve_pairwise_id(v) for v in value)
        else:
            a_key, b_key = str(key).split("_", 1)
            ids.add(_resolve_pairwise_id(a_key))
            ids.add(_resolve_pairwise_id(b_key))
    return ids


def pairwise_raw_to_embedding_payload(raw: Dict[str, Any], id_map: Dict[Any, int]) -> EmbeddingPayload:
    """Build a square pairwise `EmbeddingPayload` (item-item/user-user similarity or
    sentiment) from a dict already parsed (via `Reader.read_json`) from either of the
    two raw layouts used across Elliot's pairwise loaders -- told apart per-entry by the
    value's type, so both can even coexist in the same file:

    - adjacency-list: `{"a": ["b", "c"]}` (unweighted, edge weight defaults to `1.0`)
    - weighted-pair-key: `{"a_b": 0.42}` (`_`-joined pair key, explicit float weight)
    """
    rows, cols, data = [], [], []
    for key, value in raw.items():
        if isinstance(value, list):
            src = _resolve_pairwise_id(key)
            if src not in id_map:
                continue
            for dst_key in value:
                dst = _resolve_pairwise_id(dst_key)
                if dst not in id_map:
                    continue
                rows.append(id_map[src])
                cols.append(id_map[dst])
                data.append(1.0)
        else:
            a_key, b_key = str(key).split("_", 1)
            src, dst = _resolve_pairwise_id(a_key), _resolve_pairwise_id(b_key)
            if src not in id_map or dst not in id_map:
                continue
            rows.append(id_map[src])
            cols.append(id_map[dst])
            data.append(float(value))

    row_ids = sorted(id_map, key=id_map.get)
    sparse = sp.csr_matrix((data, (rows, cols)), shape=(len(id_map), len(id_map)))
    return EmbeddingPayload(
        sparse=sparse,
        row_ids=row_ids,
        id_map=dict(id_map),
        col_ids=row_ids,
        shape=sparse.shape
    )


def build_entity_relation_index(
    triples: List[Tuple[str, str, str]],
    reciprocal: bool = False,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int], Dict[str, int]]:
    """From a list of `(s, p, o)` string triples, build sorted (deterministic)
    `entity2id`/`relation2id` indices. When `reciprocal`, an `inverse_<predicate>`
    relation plus the reversed `(o, inverse_p, s)` triple is added for every original
    triple. Returns `(triples, entity2id, relation2id)`, where `triples` includes the
    added reciprocal triples, if any.
    """
    if reciprocal:
        triples = list(triples) + [(o, f"inverse_{p}", s) for (s, p, o) in triples]

    entities = {s for s, _, _ in triples} | {o for _, _, o in triples}
    predicates = {p for _, p, _ in triples}

    entity2id = public_id_map(entities)
    relation2id = public_id_map(predicates)
    return triples, entity2id, relation2id


def triples_to_graph_payload(
    triples: List[Tuple[str, str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    item_entity_map: Optional[Dict[Any, int]] = None,
    user_entity_map: Optional[Dict[Any, int]] = None,
) -> GraphPayload:
    """Vectorize `(s, p, o)` string triples into the canonical `GraphPayload`, using an
    existing `entity2id`/`relation2id` index (see `build_entity_relation_index`).
    """
    heads = np.array([entity2id[s] for s, _, _ in triples], dtype=np.int64)
    relations = np.array([relation2id[p] for _, p, _ in triples], dtype=np.int64)
    tails = np.array([entity2id[o] for _, _, o in triples], dtype=np.int64)
    return GraphPayload(
        heads=heads,
        relations=relations,
        tails=tails,
        entity2id=entity2id,
        relation2id=relation2id,
        item_entity_map=item_entity_map,
        user_entity_map=user_entity_map,
    )
