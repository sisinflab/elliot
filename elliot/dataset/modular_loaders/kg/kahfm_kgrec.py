from typing import List, Dict, Optional
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
class KAHFMLoader(AbstractLoader):
    """Categorical item features derived straight from raw `(subject, predicate,
    object)` KG triples (train/dev/test), unlike `ChainedKG`, which expects an
    already-flattened feature map. Every distinct `(predicate, object)` pair observed
    for a mapped item's URI becomes one feature id.
    """

    mapping: str
    kg_train: str
    kg_dev: Optional[str] = None
    kg_test: Optional[str] = None
    properties: Optional[str] = None
    additive: bool = True
    threshold: float = 1.0

    def __init__(self, **params):
        super().__init__(**params)

        # Initializing variables
        self._entity_mapping: Dict[int, str] = {}
        self._triples: pd.DataFrame = pd.DataFrame()

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
                skip_fn=lambda line: line.startswith("#")
            )

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

        if property_list:
            if self.additive:
                self._triples = self._triples[self._triples["predicate"].isin(property_list)]
            else:
                self._triples = self._triples[~self._triples["predicate"].isin(property_list)]

        self.filter_triples()

        self.items = self.items & set(self._entity_mapping.keys())

    def filter(self, users, items):
        super().filter(users, items)
        self._entity_mapping = {k: v for k, v in self._entity_mapping.items() if k in items}
        self.filter_triples()
        self.items = self.items & set(self._entity_mapping.keys())

    def filter_triples(self):
        self._triples = self._triples[self._triples["uri"].isin(self._entity_mapping.values())]
        n_mapped_subjects = self._triples["uri"].nunique()
        self._triples = (
            self._triples
            .groupby(["predicate", "object"])
            .filter(lambda x: (1 - len(x) / n_mapped_subjects) <= self.threshold)
            .astype(str)
        )
        mapped_items = [str(uri) for uri in self._triples["uri"].unique()]
        self.logger.info(
            f"Filtering operation: KAHFM Mapped items:\t{len(self.items)}"
        )
        self._entity_mapping = {
            k: v for k, v in self._entity_mapping.items()
            if v in mapped_items
        }

    def load(self) -> Dict[str, EmbeddingPayload]:
        inverted_mapping = {v: k for k, v in self._entity_mapping.items()}
        feature_list = list(
            self._triples
            .groupby(["predicate", "object"])
            .indices.keys()
        )
        self.logger.info(
            f"Final KAHFM Features:\t{len(feature_list)}\t"
            f"Mapped items:\t{len(self.items)}"
        )

        feature_index = {k: p for p, k in enumerate(feature_list)}
        self._triples["idx_feature"] = (
            self._triples[["predicate", "object"]]
            .set_index(["predicate", "object"])
            .index.map(feature_index)
        )
        map_ = self._triples.groupby("uri")["idx_feature"].apply(list).to_dict()
        map_ = {
            inverted_mapping[k]: v for k, v in map_.items()
            if k in inverted_mapping.keys()
        }

        return {"item_features": raw_feature_map_to_embedding_payload(map_, self.items)}
