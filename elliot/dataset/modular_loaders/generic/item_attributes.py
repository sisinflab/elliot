from typing import Any, Dict, List

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import raw_feature_map_to_embedding_payload
from elliot.dataset.modular_loaders.formats import EmbeddingPayload
from elliot.utils.enums import EntityAxis
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="embedding",
    entity_axis={"item_features": EntityAxis.ITEM}
)
class ItemAttributes(AbstractLoader):
    """Categorical item features from a flat `item -> [feature id, ...]` map file."""

    attribute_file: str

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._map: Dict[int, List[int]] = {}

        self._map = self.reader.read_key_value_lines(
            path=self.attribute_file,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding,
            key_fn=int,
            value_fn=lambda rest: list(set([int(x) for x in rest]))
        )

        # Narrow the item domain to items with at least one feature
        self.items = self.items & set(self._map.keys())

        self.logger.debug(
            "Loaded item attributes",
            extra={
                "context": {
                    "source": self.name,
                    "items_with_features": len(self.items),
                    "unique_features": len(set(f for feats in self._map.values() for f in feats)),
                }
            },
        )

    def load(self) -> Dict[str, EmbeddingPayload]:
        """Build the `item_features` payload from `self._map`.

        Returns:
            Dict[str, EmbeddingPayload]: The `item_features` payload.
        """
        return {"item_features": raw_feature_map_to_embedding_payload(self._map, self.items)}
