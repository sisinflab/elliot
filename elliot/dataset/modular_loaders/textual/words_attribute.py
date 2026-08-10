from typing import Dict, Any, Optional
import numpy as np

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import public_id_map
from elliot.dataset.modular_loaders.formats import EmbeddingPayload, Payload, TextPayload
from elliot.utils.enums import AlignmentMode, Materialization
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="text",
    alignment=AlignmentMode.PAD,
    materialization=Materialization.MMAP
)
class WordsTextualAttributes(AbstractLoader):
    """Tokenized text plus a shared vocabulary/token embedding table for users and/or
    items. `"tokens"` merges user *and* item ids into one `id_map` (see `load()`), and
    `"*_vocab_embeddings"` is indexed by vocabulary token, not by any user/item id --
    a model consuming this loader must do its own user/item-id translation.
    """

    users_vocabulary_features: str
    items_vocabulary_features: str
    users_tokens: str
    items_tokens: str
    pos_users: Optional[str] = None
    pos_items: Optional[str] = None

    def __init__(self, **params):
        super().__init__(**params)

        # Initializing variables
        self._users_tokens_data: Dict[int, Any] = {}
        self._items_tokens_data: Dict[int, Any] = {}
        self._word_feature_shape: int = 0
        self._pos_users_data: Dict[int, Any] = {}
        self._pos_items_data: Dict[int, Any] = {}
        self._item_mapping: Dict[int, int] = {}
        self._user_mapping: Dict[int, int] = {}

        self._users_tokens_data = {
            int(k): v for k, v in self.reader.read_json(self.users_tokens).items()
        }
        self._items_tokens_data = {
            int(k): v for k, v in self.reader.read_json(self.items_tokens).items()
        }

        # Shape-sniff via a memory-mapped read: avoids pulling the whole
        # (potentially large) vocabulary embedding matrix into memory just to
        # discover the id domain -- the full array is only materialized in load().
        self._word_feature_shape = self.reader.peek_npy_shape(
            path=self.users_vocabulary_features
        )[-1]

        if self.pos_users is not None and self.pos_items is not None:
            self._pos_users_data = {
                int(k): v for k, v in self.reader.read_json(self.pos_users).items()
            }
            self._pos_items_data = {
                int(k): v for k, v in self.reader.read_json(self.pos_items).items()
            }

        users = set(self._users_tokens_data.keys())
        items = set(self._items_tokens_data.keys())

        if users:
            self._user_mapping = public_id_map(users)
        if items:
            self._item_mapping = public_id_map(items)

        self.users = self.users & users
        self.items = self.items & items

    def load(self) -> Dict[str, Payload]:
        """Return the tokenized text (`TextPayload`, cheap -- already resident from the
        id-discovery pass in `__init__`) plus the shared vocabulary/token embedding
        table(s) (`EmbeddingPayload`, shaped by `self.materialization` via
        `_vocab_payload`).
        """
        payloads: Dict[str, Payload] = {}

        if self._users_tokens_data or self._items_tokens_data:
            tokens = {}
            if self._users_tokens_data:
                tokens.update(self._users_tokens_data)
            if self._items_tokens_data:
                tokens.update(self._items_tokens_data)
            id_map = {}
            id_map.update(self._user_mapping)
            id_map.update(self._item_mapping)
            payloads["tokens"] = TextPayload(
                tokens=tokens,
                id_map=id_map,
                vocab_size=self._word_feature_shape,
            )

        if self.users_vocabulary_features:
            payloads["users_vocab_embeddings"] = self._vocab_payload(self.users_vocabulary_features)

        if self.items_vocabulary_features:
            payloads["items_vocab_embeddings"] = self._vocab_payload(self.items_vocabulary_features)

        return payloads

    def _vocab_payload(self, path: str) -> EmbeddingPayload:
        """Build the `EmbeddingPayload` for one shared vocabulary/token embedding
        table, honoring `self.materialization`. Unlike the per-item `.npy`-folder
        loaders, there is exactly *one* file here (not one per row), so `LAZY` and
        `MMAP` both have to read it via a memory-mapped `numpy.load` -- re-reading the
        whole (potentially large) table from scratch for every single row, as plain
        `LAZY` does in the per-file case, would be pathological here. They differ
        instead in what they hand back: `MMAP` exposes the memory-mapped table
        directly as `dense` (bulk/whole-matrix access, still page-cached rather than
        copied into RAM), while `LAZY` only exposes a `row_loader` over that same
        memory-mapped table, matching the row-at-a-time access contract every other
        `LAZY` loader in this codebase uses. `MEMORY` is the only one that actually
        copies the table into a fresh in-memory array.
        """
        if self.materialization == Materialization.MEMORY:
            dense = self.reader.read_npy(path)
            return EmbeddingPayload(dense=dense, shape=dense.shape)

        memmap = self.reader.read_npy(path, mmap_mode="r")
        if self.materialization == Materialization.LAZY:
            def row_loader(row_idx, _arr=memmap):
                return np.asarray(_arr[row_idx])
            return EmbeddingPayload(row_loader=row_loader, shape=memmap.shape)

        return EmbeddingPayload(dense=memmap, shape=memmap.shape)
