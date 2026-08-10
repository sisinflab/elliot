from typing import Dict, Any

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import (
    pairwise_ids_from_raw,
    pairwise_raw_to_embedding_payload,
    public_id_map
)
from elliot.dataset.modular_loaders.formats import EmbeddingPayload
from elliot.utils.enums import EntityAxis
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="embedding",
    entity_axis={"item_similarity": EntityAxis.ITEM}
)
class ItemItem(AbstractLoader):
    interactions_ii: str

    def __init__(self, **params):
        super().__init__(**params)

        # Initializing variables
        self._raw: Dict[str, Any] = {}

        self._raw = self.reader.read_json(
            path=self.interactions_ii,
            encoding=self._reader_config.encoding
        )

        self.items = self.items & pairwise_ids_from_raw(self._raw)

    def get_all_features(self, public_items):
        rows_ii, cols_ii = [], []
        for k, v in self._raw.items():
            for val in v:
                rows_ii.append(public_items[k if not k.isdigit() else int(k)])
                cols_ii.append(public_items[val])

        return rows_ii, cols_ii

    def load(self) -> Dict[str, EmbeddingPayload]:
        id_map = public_id_map(self.items)
        payload = pairwise_raw_to_embedding_payload(self._raw, id_map)
        return {"item_similarity": payload}
