from typing import Any, Dict, List, Optional, Set
from collections import Counter

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
class ChainedKG(AbstractLoader):
    """Categorical item features from a KG already flattened, outside Elliot, into a
    plain `item -> [feature id, ...]` map plus a `feature id -> predicate URI` lookup
    (used to filter features by predicate, via `properties`). For loading straight
    from raw `(subject, predicate, object)` KG triples instead, see `KAHFMLoader`.
    """

    map: str
    features: str
    properties: Optional[str] = None
    additive: bool = True
    threshold: int = 10

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._map: Dict[int, List[int]] = {}
        self._feature_names: Dict[int, Any] = {}
        self._property_list: List[str] = []

        self._map = self.reader.read_key_value_lines(
            path=self.map,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding,
            key_fn=int,
            value_fn=lambda rest: list(set([int(x) for x in rest]))
        )

        # Strip the surrounding "<...>" and split into a [predicate, object] pattern
        def _value_fn(rest):
            pattern = rest[0].split('><')
            pattern[0] = pattern[0][1:]
            pattern[-1] = pattern[-1][:-1]
            return pattern

        self._feature_names = self.reader.read_key_value_lines(
            path=self.features,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding,
            key_fn=int,
            value_fn=_value_fn
        )

        # Optional predicate whitelist/blacklist for feature selection
        if self.properties is not None:
            self._property_list = self.reader.read_lines(
                path=self.properties,
                encoding=self._reader_config.encoding,
                skip_fn=lambda line: line.startswith("#")
            )

        self._map = self.reduce_attribute_map_property_selection()

        # Narrow the item domain to those left with at least one feature
        self.items = self.items & set(self._map.keys())

    def filter(self, users: Set[Any], items: Set[Any]):
        """See `AbstractLoader.filter`. Also re-derives `self._map` for the new
        `items` domain via `reduce_attribute_map_property_selection`.

        Args:
            users (Set[Any]): The narrowed users domain.
            items (Set[Any]): The narrowed items domain.
        """
        super().filter(users, items)
        self._map = {k: v for k, v in self._map.items() if k in self.items}
        self._map = self.reduce_attribute_map_property_selection()
        self.items = self.items & set(self._map.keys())

    def reduce_attribute_map_property_selection(self) -> Dict[int, List[int]]:
        """Restrict `self._feature_names` to the configured `properties`
        (additive/subtractive, per `self.additive`), then threshold-filter
        `self._map` to features occurring more than `self.threshold` times among
        the current `self.items`, dropping items left with no feature.

        Returns:
            Dict[int, List[int]]: The filtered `item -> [feature id, ...]` map.
        """
        acceptable_features = set()
        if not self._property_list:
            # No property filter: every feature is acceptable
            acceptable_features.update(self._feature_names.keys())
        else:
            # Keep (additive) or drop (subtractive) features whose predicate matches
            for feature in self._feature_names.items():
                if self.additive:
                    if feature[1][0] in self._property_list:
                        acceptable_features.add(int(feature[0]))
                else:
                    if feature[1][0] not in self._property_list:
                        acceptable_features.add(int(feature[0]))

        self.logger.info(
            f"Acceptable Features:\t{len(acceptable_features)}\t"
            f"Mapped items:\t{len(self._map)}"
        )

        nmap = {k: v for k, v in self._map.items() if k in self.items}

        # Count how many (current) items each acceptable feature covers
        feature_occurrences_dict = Counter([
            x for xs in nmap.values() for x in xs
            if x in acceptable_features
        ])
        features_popularity = {
            k: v for k, v in feature_occurrences_dict.items() if v > self.threshold
        }

        self.logger.info(f"Features above threshold:\t{len(features_popularity)}")

        # Drop below-threshold features, then items left with none
        new_map = {
            k: [value for value in v if value in features_popularity.keys()]
            for k, v in nmap.items()
        }
        new_map = {k: v for k, v in new_map.items() if len(v) > 0}
        self.logger.info(f"Final #items:\t{len(new_map.keys())}")

        return new_map

    def load(self) -> Dict[str, EmbeddingPayload]:
        """Build the `item_features` payload from `self._map`.

        Returns:
            Dict[str, EmbeddingPayload]: The `item_features` payload.
        """
        return {"item_features": raw_feature_map_to_embedding_payload(self._map, self.items)}
