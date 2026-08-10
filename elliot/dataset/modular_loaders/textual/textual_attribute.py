from typing import Dict, Optional

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
    files (filename stem = item id). Generic over what the vector represents --
    text/document embeddings, aspect-based sentiment embeddings, or any other
    precomputed per-item dense representation.
    """

    textual_features: str

    def __init__(self, **params):
        super().__init__(**params)

        # Initializing variables
        self._item_mapping: Dict[int, int] = {}
        self._textual_features_shape: Optional[int] = None

        inner_items, self._item_mapping, shape = self.reader.discover_npy_ids(
            folder_path=self.textual_features
        )
        if shape is not None:
            self._textual_features_shape = shape[0]

        self.items = self.items & inner_items

    def get_all_features(self):
        return self.get_all_textual_features()

    def get_all_textual_features(self):
        return embedding_to_dense(self.load()["item_features"])

    def load(self) -> Dict[str, EmbeddingPayload]:
        id_map = public_id_map(i for i in self._item_mapping if i in self.items)
        shape = self._textual_features_shape
        if shape is not None:
            shape = (shape,)
        payload = npy_folder_to_embedding_payload(
            self.textual_features, id_map, shape, self.materialization, self.reader
        )
        return {"item_features": payload}
