"""Canonical interchange formats for side-information payloads: every modular loader
converts whatever raw format it reads into one of `EmbeddingPayload`, `TextPayload`,
or `GraphPayload` before handing it to a recommender. These are plain data containers
with no behavior.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    """Raw text and/or tokenized sequences per id."""

    tokens: Optional[Dict[Any, List[int]]] = None
    raw_text: Optional[Dict[Any, str]] = None
    id_map: Optional[Dict[Any, int]] = None
    vocab_size: Optional[int] = None


@dataclass
class GraphPayload:
    """Canonical RDF-triple encoding: parallel (head, relation, tail) id arrays."""

    heads: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    relations: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    tails: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    entity2id: Dict[Any, int] = field(default_factory=dict)
    relation2id: Dict[Any, int] = field(default_factory=dict)
    item_entity_map: Optional[Dict[Any, int]] = None
    user_entity_map: Optional[Dict[Any, int]] = None


Payload = Union[EmbeddingPayload, TextPayload, GraphPayload]
