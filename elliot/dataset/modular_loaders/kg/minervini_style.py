from typing import List, Tuple, Dict, Literal, Optional

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import build_entity_relation_index, triples_to_graph_payload
from elliot.dataset.modular_loaders.formats import GraphPayload
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="kg_edges",
    format="graph"
)
class KGCompletion(AbstractLoader):
    """A standalone KG-completion dataset (train/dev/test/test_i/test_ii triple
    splits), for models that learn directly over the knowledge graph rather than over
    item/user-derived categorical features. Optionally accepts a `mapping` file (item
    id -> KG entity uri) to also expose an `item_entity_map` on the produced
    `GraphPayload`.
    """

    train_path: str
    dev_path: Optional[str] = None
    test_path: Optional[str] = None
    test_i_path: Optional[str] = None
    test_ii_path: Optional[str] = None
    input_type: Literal["standard", "reciprocal"] = "standard"
    mapping: Optional[str] = None

    def __init__(self, **params):
        super().__init__(**params)

        # Initializing variables
        self._item_mapping: Dict[int, str] = {}
        self._train_triples: List[Tuple[str, str, str]] = []
        self._dev_triples: List[Tuple[str, str, str]] = []
        self._test_triples: List[Tuple[str, str, str]] = []
        self._test_i_triples: List[Tuple[str, str, str]] = []
        self._test_ii_triples: List[Tuple[str, str, str]] = []
        self._entity_to_idx: Dict[str, int] = {}
        self._predicate_to_idx: Dict[str, int] = {}

        if self.mapping is not None:
            self._item_mapping = self.reader.read_key_value_lines(
                path=self.mapping,
                sep=self._reader_config.sep,
                encoding=self._reader_config.encoding,
                key_fn=int,
                value_fn=lambda rest: rest[0]
            )

        train_triples = self.reader.read_triples_as_tuples(
            path=self.train_path,
            encoding=self._reader_config.encoding
        )
        if self.input_type == "reciprocal":
            train_triples = train_triples + [(o, f"inverse_{p}", s) for (s, p, o) in train_triples]

        self._train_triples = train_triples

        if self.dev_path is not None:
            self._dev_triples = self.reader.read_triples_as_tuples(
                path=self.dev_path,
                encoding=self._reader_config.encoding
            )
        if self.test_path is not None:
            self._test_triples = self.reader.read_triples_as_tuples(
                path=self.test_path,
                encoding=self._reader_config.encoding
            )
        if self.test_i_path is not None:
            self._test_i_triples = self.reader.read_triples_as_tuples(
                path=self.test_i_path,
                encoding=self._reader_config.encoding
            )
        if self.test_ii_path is not None:
            self._test_ii_triples = self.reader.read_triples_as_tuples(
                path=self.test_ii_path,
                encoding=self._reader_config.encoding
            )

        all_triples = train_triples + self._dev_triples + self._test_triples
        _, self._entity_to_idx, self._predicate_to_idx = build_entity_relation_index(
            all_triples, reciprocal=False
        )

        inverse_of_idx = {}
        original_predicates = {p for _, p, _ in train_triples}
        if self.input_type == "reciprocal":
            for p in original_predicates:
                p_idx, ip_idx = self._predicate_to_idx[p], self._predicate_to_idx[f"inverse_{p}"]
                inverse_of_idx.update({p_idx: ip_idx, ip_idx: p_idx})

    def load(self) -> Dict[str, GraphPayload]:
        item_entity_map = None
        if self._item_mapping:
            item_entity_map = {
                item: self._entity_to_idx[uri]
                for item, uri in self._item_mapping.items()
                if item in self.items and uri in self._entity_to_idx
            }

        payloads = {
            "kg_triples": triples_to_graph_payload(
                self._train_triples, self._entity_to_idx, self._predicate_to_idx, item_entity_map
            )
        }
        if self._dev_triples:
            payloads["kg_dev_triples"] = triples_to_graph_payload(
                self._dev_triples, self._entity_to_idx, self._predicate_to_idx
            )
        if self._test_triples:
            payloads["kg_test_triples"] = triples_to_graph_payload(
                self._test_triples, self._entity_to_idx, self._predicate_to_idx
            )

        return payloads
