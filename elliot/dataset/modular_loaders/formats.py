"""Canonical interchange formats for side-information payloads: every modular loader
converts whatever raw format it reads into one of `EmbeddingPayload`, `TextPayload`,
or `GraphPayload` before handing it to a recommender. These are plain data containers
with no behavior.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp


@dataclass
class EmbeddingPayload:
    """Dense/sparse per-id numeric vectors, or a pairwise id x id matrix.

    Exactly one of `dense`, `sparse`, `row_loader` is expected to be populated:

    - `dense`/`sparse`: the whole matrix is already materialized in memory
      (`Materialization.MEMORY`).
    - `row_loader`: an on-demand accessor `row_index -> np.ndarray`, used instead of
      materializing the whole matrix when the loader declares `Materialization.LAZY`/
      `MMAP` (e.g. one `.npy` file per item, read on demand).

    `row_ids` lists the domain identifiers (item/user/token ids) in row order; `id_map`
    maps a domain identifier to its row index. `col_ids` optionally names what each
    column represents (a feature id, or the paired entity for an id x id matrix).

    Attributes:
        dense (np.ndarray, optional): Fully materialized dense matrix. Defaults to None.
        sparse (sp.spmatrix, optional): Fully materialized sparse matrix. Defaults to None.
        row_loader (Callable[[int], np.ndarray], optional): On-demand accessor
            `row_index -> np.ndarray`. Defaults to None.
        row_ids (List[Any], optional): The domain ids (item/user/token ids), in row
            order. Defaults to None.
        id_map (Dict[Any, int], optional): Mapping from a domain id to its row index.
            Defaults to None.
        col_ids (List[Any], optional): What each column represents (a feature id, or
            the paired entity for an id x id matrix). Defaults to None.
        shape (Tuple[int, int], optional): Shape of the payload. Defaults to None.
    """

    dense: Optional[np.ndarray] = None
    sparse: Optional[sp.spmatrix] = None
    row_loader: Optional[Callable[[int], np.ndarray]] = None
    row_ids: Optional[List[Any]] = None
    id_map: Optional[Dict[Any, int]] = None
    col_ids: Optional[List[Any]] = None
    shape: Optional[Tuple[int, int]] = None


@dataclass
class TextPayload:
    """Raw text and/or tokenized sequences per id.

    Attributes:
        tokens (Dict[Any, List[int]], optional): Tokenized sequence per id.
            Defaults to None.
        raw_text (Dict[Any, str], optional): Raw text per id. Defaults to None.
        id_map (Dict[Any, int], optional): Mapping from a domain id to its row index.
            Defaults to None.
        vocab_size (int, optional): Size of the shared vocabulary the tokens are
            drawn from. Defaults to None.
    """

    tokens: Optional[Dict[Any, List[int]]] = None
    raw_text: Optional[Dict[Any, str]] = None
    id_map: Optional[Dict[Any, int]] = None
    vocab_size: Optional[int] = None


@dataclass
class GraphPayload:
    """Canonical RDF-triple encoding: parallel (head, relation, tail) id arrays.

    Attributes:
        heads (np.ndarray): Head entity id per triple.
        relations (np.ndarray): Relation id per triple.
        tails (np.ndarray): Tail entity id per triple.
        entity2id (Dict[Any, int]): Entity id index (see `build_entity_relation_index`).
        relation2id (Dict[Any, int]): Relation id index (see
            `build_entity_relation_index`).
        item_entity_map (Dict[Any, int], optional): Item id -> KG entity id map.
            Defaults to None.
        user_entity_map (Dict[Any, int], optional): User id -> KG entity id map.
            Defaults to None.
    """

    heads: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    relations: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    tails: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    entity2id: Dict[Any, int] = field(default_factory=dict)
    relation2id: Dict[Any, int] = field(default_factory=dict)
    item_entity_map: Optional[Dict[Any, int]] = None
    user_entity_map: Optional[Dict[Any, int]] = None


Payload = Union[EmbeddingPayload, TextPayload, GraphPayload]
