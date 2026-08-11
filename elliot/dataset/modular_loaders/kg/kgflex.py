from typing import Any, Dict, List, Optional, Set
import pandas as pd

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
class KGFlexLoader(AbstractLoader):
    """Categorical item features from raw KG triples, like `KAHFMLoader`, but also
    mines 2-hop path features: every `(predicate_x, predicate_y, object_y)` path
    reached via an item's first-hop object into a second KG (`second_hop`) becomes an
    extra feature id, unioned with the 1-hop features into the same `item_features`
    payload.
    """

    mapping: str
    kg_train: str
    kg_dev: Optional[str] = None
    kg_test: Optional[str] = None
    second_hop: Optional[str] = None
    properties: Optional[str] = None
    additive: bool = True
    threshold: int = 10

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._entity_mapping: Dict[int, str] = {}
        self._triples: pd.DataFrame = pd.DataFrame()
        self._second_hop_triples: pd.DataFrame = (
            pd.DataFrame(columns=["uri", "predicate", "object"])
            .astype(dtype={"uri": str, "predicate": str, "object": str})
        )
        self._second_order_features = (
            pd.DataFrame(columns=["uri_x", "predicate_x", "object_x", "predicate_y", "object_y"])
            .astype(dtype={
                "uri_x": str,
                "predicate_x": str,
                "object_x": str,
                "predicate_y": str,
                "object_y": str
            })
        )

        self._entity_mapping = self.reader.read_key_value_lines(
            path=self.mapping,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding,
            key_fn=int,
            value_fn=lambda rest: rest[0]
        )

        property_list: List[str] = []
        if self.properties is not None:
            property_list = self.reader.read_lines(
                path=self.properties,
                encoding=self._reader_config.encoding,
                skip_fn=lambda line: line.startswith('#')
            )

        # Load and merge every configured KG split into one triples table
        train_triples = self.reader.read_triples(
            path=self.kg_train,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding
        )

        dev_triples: pd.DataFrame = pd.DataFrame()
        test_triples: pd.DataFrame = pd.DataFrame()
        if self.kg_dev is not None:
            dev_triples = self.reader.read_triples(
                path=self.kg_dev,
                sep=self._reader_config.sep,
                encoding=self._reader_config.encoding
            )
        if self.kg_test is not None:
            test_triples = self.reader.read_triples(
                path=self.kg_test,
                sep=self._reader_config.sep,
                encoding=self._reader_config.encoding
            )

        self._triples = pd.concat([train_triples, dev_triples, test_triples])
        del train_triples, dev_triples, test_triples

        # Optionally load the second KG hop used for 2-hop path features
        if self.second_hop is not None:
            self._second_hop_triples = self.reader.read_triples(
                path=self.second_hop,
                sep=self._reader_config.sep,
                encoding=self._reader_config.encoding
            )

        # Keep (additive) or drop (subtractive) triples matching the property list, in both hops
        if property_list:
            if self.additive:
                self._triples = self._triples[self._triples["predicate"].isin(property_list)]
                self._second_hop_triples = self._second_hop_triples[
                    self._second_hop_triples["predicate"].isin(property_list)
                ]
            else:
                self._triples = self._triples[~self._triples["predicate"].isin(property_list)]
                self._second_hop_triples = self._second_hop_triples[
                    ~self._second_hop_triples["predicate"].isin(property_list)
                ]

        self._compute_features()

        # Narrow the entity mapping to items that survived feature mining
        possible_items = [str(uri) for uri in self._triples["uri"].unique()]
        self._entity_mapping = {
            k: v for k, v in self._entity_mapping.items()
            if v in possible_items
        }

        self.items = self.items & set(self._entity_mapping.keys())

    def filter(self, users: Set[Any], items: Set[Any]):
        """See `AbstractLoader.filter`. Also drops entity-mapping entries outside the
        new `items` domain.

        Args:
            users (Set[Any]): The narrowed users domain.
            items (Set[Any]): The narrowed items domain.
        """
        super().filter(users, items)
        self._entity_mapping = {k: v for k, v in self._entity_mapping.items() if k in items}
        self.items = {i for i in self.items if i in self._entity_mapping.keys()}

    def _compute_features(self):
        """Threshold-filter 1-hop `(predicate, object)` and 2-hop `(predicate_x,
        predicate_y, object_y)` features by occurrence count, mirroring the original
        KGFlex feature-mining logic.
        """
        # Threshold-filter 1-hop (predicate, object) features by occurrence count
        occurrences_per_feature = self._triples.groupby(["predicate", "object"]).size().to_dict()
        keep_set = {
            f for f, occ in occurrences_per_feature.items()
            if occ > self.threshold
        }

        # Join each item's first-hop object into the second KG, to mine 2-hop paths
        second_order = self._triples.merge(
            self._second_hop_triples, left_on="object", right_on="uri", how="left"
        )
        second_order = second_order[second_order["uri_y"].notna()]

        # Threshold-filter 2-hop (predicate_x, predicate_y, object_y) features too
        occurrences_per_feature_2 = (
            second_order
            .groupby(["predicate_x", "predicate_y", "object_y"])
            .size().to_dict()
        )
        keep_set2 = {
            f for f, occ in occurrences_per_feature_2.items()
            if occ > self.threshold
        }

        # Keep only 1-hop triples whose feature survived the threshold
        self._triples = self._triples[
            self._triples[["predicate", "object"]]
            .set_index(["predicate", "object"])
            .index.map(lambda f: f in keep_set)
        ].astype(str)

        # Keep only 2-hop paths whose feature survived the threshold
        if len(second_order) > 0:
            self._second_order_features = (second_order[
                second_order[["predicate_x", "predicate_y", "object_y"]]
                .set_index(["predicate_x", "predicate_y", "object_y"])
                .index.map(lambda f: f in keep_set2)
            ].astype(str))
            self._second_order_features = self._second_order_features.drop(["uri_y"], axis=1)

    def load(self) -> Dict[str, EmbeddingPayload]:
        """Build the `item_features` payload, unioning 1-hop `(predicate, object)`
        features with 2-hop `(predicate_x, predicate_y, object_y)` path features
        (offset past the 1-hop feature ids) into a single feature space.

        Returns:
            Dict[str, EmbeddingPayload]: The `item_features` payload.
        """
        inverted_mapping = {v: k for k, v in self._entity_mapping.items()}

        # Assign a contiguous index to every distinct 1-hop feature
        first_keys = list(self._triples.groupby(["predicate", "object"]).indices.keys())
        first_index = {k: i for i, k in enumerate(first_keys)}

        triples = self._triples.copy()
        triples["idx_feature"] = (
            triples[["predicate", "object"]]
            .set_index(["predicate", "object"])
            .index.map(first_index)
        )
        first_map = triples.groupby("uri")["idx_feature"].apply(list).to_dict()

        second_map = {}
        if len(self._second_order_features):
            # Assign a contiguous index to every distinct 2-hop feature, offset past 1-hop ids
            second_keys = list(
                self._second_order_features
                .groupby(["predicate_x", "predicate_y", "object_y"])
                .indices.keys()
            )
            second_index = {k: i + len(first_index) for i, k in enumerate(second_keys)}

            second = self._second_order_features.copy()
            second["idx_feature"] = (
                second[["predicate_x", "predicate_y", "object_y"]]
                .set_index(["predicate_x", "predicate_y", "object_y"])
                .index.map(second_index)
            )
            second_map = second.groupby("uri_x")["idx_feature"].apply(list).to_dict()

        # Union 1-hop and 2-hop feature lists per item, then translate uri -> item id
        combined = {
            uri: first_map.get(uri, []) + second_map.get(uri, [])
            for uri in set(first_map) | set(second_map)
        }
        feature_map = {
            inverted_mapping[uri]: feats for uri, feats in combined.items()
            if uri in inverted_mapping
        }

        return {"item_features": raw_feature_map_to_embedding_payload(feature_map, self.items)}
