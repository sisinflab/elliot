from typing import Any, Dict, Optional
import numpy as np

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import npy_folder_to_embedding_payload, public_id_map
from elliot.dataset.modular_loaders.materialize import embedding_to_dense
from elliot.dataset.modular_loaders.formats import EmbeddingPayload
from elliot.utils.enums import AlignmentMode, EntityAxis, Materialization
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="embedding",
    alignment=AlignmentMode.PAD,
    materialization=Materialization.LAZY,
    entity_axis={"item_features": EntityAxis.ITEM}
)
class TextualAttribute(AbstractLoader):
    """One precomputed dense feature vector per item, read from a folder of `.npy`
    files (filename stem = item id). Generic over what the vector represents -
    text/document embeddings, aspect-based sentiment embeddings, or any other
    precomputed per-item dense representation.
    """

    textual_features: str

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._item_mapping: Dict[int, int] = {}
        self._textual_features_shape: Optional[int] = None

        inner_items, self._item_mapping, shape = self.reader.discover_npy_ids(
            folder_path=self.textual_features
        )
        if shape is not None:
            self._textual_features_shape = shape[0]

        # Narrow the item domain to those with a feature file
        self.items = self.items & inner_items

    def get_all_features(self) -> np.ndarray:
        """Alias for `get_all_textual_features`.

        Returns:
            np.ndarray: The dense item-features matrix.
        """
        return self.get_all_textual_features()

    def get_all_textual_features(self) -> np.ndarray:
        """Dense `(n_items, dim)` matrix of textual features, read through `load()`
        so it honors `self.materialization`.

        Returns:
            np.ndarray: The dense item-features matrix.
        """
        return embedding_to_dense(self.load()["item_features"])

    def load(self) -> Dict[str, EmbeddingPayload]:
        """Build the `item_features` payload from the folder of `.npy` files, via the
        shared `npy_folder_to_embedding_payload` adapter (honors `self.materialization`
        there, not here).

        Returns:
            Dict[str, EmbeddingPayload]: The `item_features` payload.
        """
        # Re-index to the current (post-filter) item domain
        id_map = public_id_map(i for i in self._item_mapping if i in self.items)
        shape = self._textual_features_shape
        if shape is not None:
            shape = (shape,)

        payload = npy_folder_to_embedding_payload(
            self.textual_features, id_map, shape, self.materialization, self.reader
        )
        return {"item_features": payload}
