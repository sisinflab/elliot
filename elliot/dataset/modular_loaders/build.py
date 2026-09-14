"""Raw data -> canonical payload: turns already-read raw data (a folder listing, a
parsed JSON dict, a list of KG triples, ...) into one of the canonical payloads (see
`elliot.dataset.modular_loaders.formats`). Every function here is a pure, in-memory
transform; reading the raw file/folder itself is `elliot.utils.read.Reader`'s job.
"""

from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import scipy.sparse as sp

from elliot.dataset.modular_loaders.materialize import feature_map_to_sparse
from elliot.dataset.modular_loaders.formats import EmbeddingPayload, GraphPayload
from elliot.utils.enums import Materialization
from elliot.utils.folder import path_joiner
from elliot.utils.read import Reader


def public_id_map(ids: Iterable[Any], priority: Optional[Iterable[Any]] = None) -> Dict[Any, int]:
    """Assign a deterministic `0..n-1` row index to every id in `ids`. With no
    `priority`, ids are indexed in plain sorted order. With `priority`, the ids in
    it (deduped, order preserved, anything not in `ids` dropped) get the lowest
    indices first, in that order; every other id follows, in sorted order - e.g. so
    an item's KG entity always gets a lower id than any other KG entity, the
    convention KGIN's own dataset dumps use (see `kg.kg_triples.KGTriplesLoader`).

    Args:
        ids (Iterable[Any]): The domain ids to index.
        priority (Iterable[Any], optional): Ids to place first, in this order.
            Defaults to None, indexing every id in plain sorted order.

    Returns:
        Dict[Any, int]: Mapping from a domain id to its row index.
    """
    ids = set(ids)
    if not priority:
        ids = ids if all(isinstance(x, str) for x in ids) else sorted(ids)
        return {entity_id: idx for idx, entity_id in enumerate(ids)}

    priority_ids = list(dict.fromkeys(i for i in priority if i in ids))
    remaining_ids = sorted(ids - set(priority_ids))

    id_map = {entity_id: idx for idx, entity_id in enumerate(priority_ids)}
    id_map.update({entity_id: idx + len(priority_ids) for idx, entity_id in enumerate(remaining_ids)})
    return id_map


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
    until a row is actually accessed - `LAZY` with `mmap_mode=None` (a plain read,
    copied fresh every access, no lingering file handle), `MMAP` with `mmap_mode="r"`
    (memory-mapped, read-only, backed by the OS page cache across repeated accesses to
    the same id).

    Args:
        materialization (Materialization): Materialization strategy controlling
            whether the payload is fully loaded into memory (`MEMORY`), read on
            demand (`LAZY`), or memory-mapped on demand (`MMAP`).
        id_map (Dict[Any, int]): Mapping from a domain id to its row index.
        shape (Tuple[int, ...], optional): Per-row shape of the payload (excluding
            the leading id dimension), or None if unknown.
        row_reader (Callable[[Any, Optional[str]], np.ndarray]): Callable returning
            the raw row for a given id, honoring `mmap_mode` (`None` for a plain
            read, `"r"` for a memory-mapped one) when the underlying source supports it.

    Returns:
        EmbeddingPayload: The payload, materialized or on-demand per `materialization`.
    """
    row_ids = sorted(id_map, key=id_map.get)
    full_shape = (len(id_map),) + tuple(shape) if shape else (len(id_map),)

    # MEMORY: eagerly read every row into a single dense matrix
    if materialization == Materialization.MEMORY:
        dense = np.empty(full_shape)
        for entity_id, row in id_map.items():
            dense[row] = row_reader(entity_id, None)
        return EmbeddingPayload(
            dense=dense, row_ids=row_ids, id_map=dict(id_map), shape=dense.shape
        )

    # LAZY/MMAP: defer each row read until it's actually accessed
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
    `Reader.discover_npy_ids`, which a loader uses to build `id_map`/ `shape` ahead of this call).
    A thin `row_reader` over the shared `rows_to_embedding_payload` dispatch; see that function
    for how `materialization` is actually handled. Every actual file read goes through `reader`
    (a fresh, default `Reader` if the caller doesn't already have one).

    Args:
        folder_path (str): Path to the folder holding one `.npy` file per id
            (filename stem = id).
        id_map (Dict[int, int]): Mapping from a domain id to its row index.
        shape (Tuple[int, ...], optional): Per-row shape of the payload (excluding
            the leading id dimension), or None if unknown.
        materialization (Materialization): Materialization strategy controlling
            whether the payload is fully loaded into memory (`MEMORY`), read on
            demand (`LAZY`), or memory-mapped on demand (`MMAP`). Defaults to `MMAP`.
        reader (Reader, optional): Reader instance used for the actual file access.
            Defaults to None, building a fresh default one.

    Returns:
        EmbeddingPayload: The payload, materialized or on-demand per `materialization`.
    """
    # Fall back to a fresh default Reader if the caller doesn't already have one
    reader = reader or Reader()

    def row_reader(entity_id, mmap_mode, _folder=folder_path, _reader=reader):
        return _reader.read_npy(path_joiner(_folder, f"{entity_id}.npy"), mmap_mode=mmap_mode)

    return rows_to_embedding_payload(materialization, id_map, shape, row_reader)


def raw_feature_map_to_embedding_payload(feature_map: Dict[Any, List[Any]], items: Iterable[Any]) -> EmbeddingPayload:
    """Build a categorical multi-hot `EmbeddingPayload` from a raw `dict[id -> list[feature
    id]]` whose feature ids are not yet a contiguous `0..n` column index (this function
    assigns that index itself) - the shape produced by `ItemAttributes`, `ChainedKG`,
    `KAHFMLoader` and `KGFlexLoader` alike, whichever raw KG/attribute format each reads.

    Args:
        feature_map (Dict[Any, List[Any]]): Mapping from an id to its list of (not
            yet contiguous) feature ids.
        items (Iterable[Any]): The domain ids defining the payload's rows.

    Returns:
        EmbeddingPayload: The categorical multi-hot payload.
    """
    id_map = public_id_map(items)
    row_ids = sorted(id_map, key=id_map.get)

    # Assign a contiguous 0..n-1 column index to every distinct feature id
    features = sorted({f for i in row_ids for f in feature_map.get(i, [])})
    public_features = {f: idx for idx, f in enumerate(features)}

    # Translate each row's raw feature ids into the new contiguous column index
    translated_map = {i: [public_features[f] for f in feature_map.get(i, [])] for i in row_ids}
    sparse = feature_map_to_sparse(translated_map, id_map, len(features))

    return EmbeddingPayload(
        sparse=sparse,
        row_ids=row_ids,
        id_map=id_map,
        col_ids=features,
        shape=sparse.shape
    )


def coerce_id(key: Any) -> Any:
    """Coerce a raw string (or JSON key/value) into the same id type
    `Reader.read_mapping`-style id sets use: `int` when possible (even via a
    `"1.0"`-style float string), else the original string. Shared by every reader that
    doesn't know upfront whether the raw ids it's parsing are numeric (matching an
    existing `int`-typed user/item domain) or opaque strings (e.g. KG URIs) -
    `pairwise_raw_to_embedding_payload`, `pairwise_ids_from_raw`, and the KG triples
    loaders (`kg.kg_triples`).

    Args:
        key (Any): The raw JSON key/value or string token to coerce.

    Returns:
        Any: The coerced id, as `int` when possible, else the original string.
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
    *without* building the sparse matrix - just enough to let a loader know
    its own users/items domain ahead of the (potentially much larger) `load()` pass.

    Args:
        raw (Dict[str, Any]): Raw pairwise dict, already parsed via `Reader.read_json`,
            in either of the two supported layouts (see `pairwise_raw_to_embedding_payload`).

    Returns:
        Set[Any]: The full id domain referenced by `raw`.
    """
    ids: Set[Any] = set()
    for key, value in raw.items():
        # Adjacency-list entry: collect the source id and every target id
        if isinstance(value, list):
            ids.add(coerce_id(key))
            ids.update(coerce_id(v) for v in value)

        # Weighted-pair-key entry: split "a_b" into its two ids
        else:
            a_key, b_key = str(key).split("_", 1)
            ids.add(coerce_id(a_key))
            ids.add(coerce_id(b_key))

    return ids


def pairwise_raw_to_embedding_payload(raw: Dict[str, Any], id_map: Dict[Any, int]) -> EmbeddingPayload:
    """Build a square pairwise `EmbeddingPayload` (item-item/user-user similarity or
    sentiment) from a dict already parsed (via `Reader.read_json`) from either of the
    two raw layouts used across Elliot's pairwise loaders - told apart per-entry by the
    value's type, so both can even coexist in the same file:

    - adjacency-list: `{"a": ["b", "c"]}` (unweighted, edge weight defaults to `1.0`)
    - weighted-pair-key: `{"a_b": 0.42}` (`_`-joined pair key, explicit float weight)

    Args:
        raw (Dict[str, Any]): Raw pairwise dict, already parsed via `Reader.read_json`,
            in either of the two supported layouts.
        id_map (Dict[Any, int]): Mapping from a domain id to its row index.

    Returns:
        EmbeddingPayload: The square pairwise payload.
    """
    rows, cols, data = [], [], []
    for key, value in raw.items():
        # Adjacency-list entry: unweighted edges, weight defaults to 1.0
        if isinstance(value, list):
            src = coerce_id(key)
            if src not in id_map:
                continue
            for dst_key in value:
                dst = coerce_id(dst_key)
                if dst not in id_map:
                    continue
                rows.append(id_map[src])
                cols.append(id_map[dst])
                data.append(1.0)

        # Weighted-pair-key entry: explicit float weight
        else:
            a_key, b_key = str(key).split("_", 1)
            src, dst = coerce_id(a_key), coerce_id(b_key)
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
    priority_entities: Optional[Iterable[Any]] = None,
) -> Tuple[List[Tuple[str, str, str]], Dict[str, int], Dict[str, int]]:
    """From a list of `(s, p, o)` string triples, build sorted (deterministic)
    `entity2id`/`relation2id` indices. When `reciprocal`, an `inverse_<predicate>`
    relation plus the reversed `(o, inverse_p, s)` triple is added for every original
    triple.

    Args:
        triples (List[Tuple[str, str, str]]): The `(subject, predicate, object)`
            string triples.
        reciprocal (bool): If True, add an `inverse_<predicate>` relation plus the
            reversed triple for every original triple. Defaults to False.
        priority_entities (Iterable[Any], optional): Entities to assign the lowest
            ids first, in this order (see `public_id_map`) - e.g. an item's KG
            entity, so it lands before any other KG entity. Defaults to None,
            indexing every entity in plain sorted order.

    Returns:
        Tuple[List[Tuple[str, str, str]], Dict[str, int], Dict[str, int]]:
            `(triples, entity2id, relation2id)`, where `triples` includes the added
            reciprocal triples, if any.
    """
    # Add the reversed inverse-relation triple for every original one
    if reciprocal:
        triples = list(triples) + [(o, f"inverse_{p}", s) for (s, p, o) in triples]

    # Collect every distinct entity (as head or tail) and relation
    entities = {s for s, _, _ in triples} | {o for _, _, o in triples}
    predicates = {p for _, p, _ in triples}

    entity2id = public_id_map(entities, priority=priority_entities)
    relation2id = public_id_map(predicates)
    return triples, entity2id, relation2id


def triples_to_graph_payload(
    triples: List[Tuple[str, str, str]],
    entity2id: Dict[str, int],
    relation2id: Dict[str, int],
    item_entity_map: Dict[Any, int],
    remap_entity: bool = True,
    remap_relation: bool = True
) -> GraphPayload:
    """Vectorize `(s, p, o)` string triples into the canonical `GraphPayload`, using an
    existing `entity2id`/`relation2id` index (see `build_entity_relation_index`).

    Args:
        triples (List[Tuple[str, str, str]]): The `(subject, predicate, object)`
            string triples.
        entity2id (Dict[str, int]): Existing entity id index (see
            `build_entity_relation_index`), keyed by whatever id `triples`
            themselves use - used here only to vectorize them.
        relation2id (Dict[str, int]): Existing relation id index, analogous to
            `entity2id`.
        item_entity_map (Dict[Any, int]): Item id -> KG entity id map, exposed on
            the produced payload as-is.
        remap_entity (bool): If True, vectorize the triples' subject/object strings 
            into the `entity2id` index. Defaults to True.
        remap_relation (bool): If True, vectorize the triples' predicate strings 
            into the `relation2id` index. Defaults to True.

    Returns:
        GraphPayload: The vectorized triples.
    """
    heads = [entity2id[s] if remap_entity else s for s, _, _ in triples]
    relations = [relation2id[p] if remap_relation else p for _, p, _ in triples]
    tails = [entity2id[o] if remap_entity else o for _, _, o in triples]

    # Vectorize each triple's (head, relation, tail) into parallel int-id arrays
    heads = np.array(heads, dtype=np.int64)
    relations = np.array(relations, dtype=np.int64)
    tails = np.array(tails, dtype=np.int64)

    # Build final id -> raw KG entity id mapping only if raw KG entity ids are provided
    id2entity = (
        None if all(coerce_id(raw_id) == coerce_id(final_id) for raw_id, final_id in entity2id.items())
        else {final_id: raw_id for raw_id, final_id in entity2id.items()}
    )
    id2relation = (
        None if all(coerce_id(raw_id) == coerce_id(final_id) for raw_id, final_id in relation2id.items())
        else {final_id: raw_id for raw_id, final_id in relation2id.items()}
    )

    return GraphPayload(
        heads=heads,
        relations=relations,
        tails=tails,
        id2entity=id2entity,
        id2relation=id2relation,
        n_entities=len(entity2id),
        n_relations=len(relation2id),
        item_entity_map=item_entity_map
    )
