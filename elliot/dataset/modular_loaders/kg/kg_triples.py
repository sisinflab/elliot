from typing import Any, Dict, List, Optional, Set, Tuple

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import build_entity_relation_index, coerce_id, triples_to_graph_payload
from elliot.dataset.modular_loaders.formats import GraphPayload
from elliot.utils.enums import EntityAxis
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="kg_edges",
    format="graph",
    entity_axis={
        "kg_triples": EntityAxis.ITEM,
        "kg_dev_triples": EntityAxis.ITEM,
        "kg_test_triples": EntityAxis.ITEM,
        "kg_test_i_triples": EntityAxis.ITEM,
        "kg_test_ii_triples": EntityAxis.ITEM,
    }
)
class KGTriplesLoader(AbstractLoader):
    """Knowledge-graph `(head, relation, tail)` triples, read from one required
    (`kg_train`) and up to four optional (`kg_dev`/`kg_test`/`kg_test_i`/`kg_test_ii`)
    whitespace- or tab-separated files, and vectorized into the canonical
    `GraphPayload` (the vectorization index is always built fresh from the observed
    triples, every item's KG entity placed first - see `build_entity_relation_index`).
    """

    kg_train: str
    kg_dev: Optional[str] = None
    kg_test: Optional[str] = None
    kg_test_i: Optional[str] = None
    kg_test_ii: Optional[str] = None
    reciprocal: bool = False
    item_mapping: Optional[str] = None
    entity_mapping: Optional[str] = None
    relation_mapping: Optional[str] = None

    def __init__(self, **params: Any):
        super().__init__(**params)

        self._train_triples = self._read_triples(self.kg_train)
        if self.reciprocal:
            self._train_triples = self._train_triples + [
                (o, f"inverse_{p}", s) for (s, p, o) in self._train_triples
            ]
        self._dev_triples = self._read_triples(self.kg_dev) if self.kg_dev else []
        self._test_triples = self._read_triples(self.kg_test) if self.kg_test else []
        self._test_i_triples = self._read_triples(self.kg_test_i) if self.kg_test_i else []
        self._test_ii_triples = self._read_triples(self.kg_test_ii) if self.kg_test_ii else []

        # raw KG id -> kg_train's own id, or None when kg_train already uses raw KG ids
        self._entity2id = (
            self._read_mapping(self.entity_mapping)
            if self.entity_mapping else None
        )
        self._relation2id = (
            self._read_mapping(self.relation_mapping)
            if self.relation_mapping else None
        )

        all_triples = self._train_triples + self._dev_triples + self._test_triples
        entities = {s for s, _, _ in all_triples} | {o for _, _, o in all_triples}

        self._item_raw_ids = self._raw_domain_ids(self.item_mapping, self.items, entities)

        self.items = self.items & self._item_raw_ids.keys()

    def _read_triples(self, path: str) -> List[Tuple[Any, Any, Any]]:
        """Read one triples file into coerced `(head, relation, tail)` tuples.

        Args:
            path (str): Path to the triples file.

        Returns:
            List[Tuple[Any, Any, Any]]: The triples, in file order, every id coerced via `coerce_id`.
        """
        raw = self.reader.read_triples_as_tuples(path=path, encoding=self._reader_config.encoding)
        return [(coerce_id(s), p, coerce_id(o)) for s, p, o in raw]

    def _read_mapping(self, path: str) -> Dict[Any, Any]:
        """Read a two-column `key <sep> value` mapping file (`item_mapping`/
        `user_mapping`/`entity_mapping`), every id coerced via `coerce_id`.

        Args:
            path (str): Path to the mapping file.

        Returns:
            Dict[Any, Any]: The `key -> value` mapping.
        """
        return self.reader.read_key_value_lines(
            path=path,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding,
            key_fn=coerce_id,
            value_fn=lambda rest: coerce_id(rest[0])
        )

    def _raw_domain_ids(
        self,
        mapping_path: Optional[str],
        domain: Set[Any],
        entities: Set[Any]
    ) -> Dict[Any, Any]:
        """Bridge `domain` (`self.items`) to a raw KG entity id actually observed
        among `entities`, reading `mapping_path` (`domain id -> raw KG id`) and,
        when `entity_mapping` was given, composing its output through
        `self._entity2id` (`raw KG id -> kg_train's own id`) first.

        Args:
            mapping_path (str, optional): Path to the `item_mapping` file, or None
                to fall back to the identity bridge.
            domain (Set[Any]): The domain ids to bridge (`self.items`).
            entities (Set[Any]): The raw ids actually observed in `kg_train` (and any
                configured split).

        Returns:
            Dict[Any, Any]: The `domain id -> raw KG id` map (only for domain ids
                that resolved to an observed entity), every domain id kept in
                `domain`'s own type (whatever `item_id_type` made it), every raw KG
                id in `coerce_id`'s canonical form (matching `entities`).
        """
        coerced_domain = {coerce_id(domain_id): domain_id for domain_id in domain}

        if mapping_path is not None:
            raw = self._read_mapping(mapping_path)
            raw_map = {
                coerced_domain[key]: raw_id for key, raw_id in raw.items()
                if key in coerced_domain
            }
        else:
            raw_map = {domain_id: coerce_id(domain_id) for domain_id in domain}

        if self._entity2id is not None:
            raw_map = {
                domain_id: self._entity2id[raw_id] for domain_id, raw_id in raw_map.items()
                if raw_id in self._entity2id
            }
        bridged = {
            domain_id: raw_id for domain_id, raw_id in raw_map.items()
            if raw_id in entities
        }

        if mapping_path is None and not bridged:
            self.logger.warning(
                "No item_mapping was given, and no domain id doubles as one of "
                "kg_train's own entities either - no item will be bridged to a KG entity."
            )

        return bridged

    def filter(self, users: Set[Any], items: Set[Any]):
        """See `AbstractLoader.filter`. Also re-derives the item/user domain against
        the (unchanged) `_item_raw_ids`/`_user_raw_ids` for the narrower domain.

        Args:
            users (Set[Any]): The narrowed users domain.
            items (Set[Any]): The narrowed items domain.
        """
        super().filter(users, items)
        filtered_item_raw_ids = {
            domain_id: raw_id
            for domain_id, raw_id in self._item_raw_ids.items()
            if domain_id in self.items
        }
        self._item_raw_ids = filtered_item_raw_ids

    def load(self) -> Dict[str, GraphPayload]:
        """Build the `kg_triples` payload (train triples, plus the item entity
        bridge), plus `kg_dev_triples`/`kg_test_triples`/`kg_test_i_triples`/
        `kg_test_ii_triples` when the corresponding split was configured.

        Returns:
            Dict[str, GraphPayload]: The KG triples payload(s).
        """
        all_triples = self._train_triples + self._dev_triples + self._test_triples
        priority = sorted(list(self._item_raw_ids.values()))
        _, entity2id, relation2id = build_entity_relation_index(all_triples, priority_entities=priority)

        item_entity_map = {
            domain_id: entity2id[raw_id]
            for domain_id, raw_id in self._item_raw_ids.items()
            if raw_id in entity2id
        }

        kwargs = {
            "entity2id": self._entity2id if self._entity2id is not None else entity2id,
            "relation2id": self._relation2id if self._relation2id is not None else relation2id,
            "item_entity_map": item_entity_map,
            "remap_entity": self._entity2id is None,
            "remap_relation": self._relation2id is None
        }

        payloads = {
            "kg_triples": triples_to_graph_payload(triples=self._train_triples, **kwargs)
        }
        for name, triples in (
            ("kg_dev_triples", self._dev_triples),
            ("kg_test_triples", self._test_triples),
            ("kg_test_i_triples", self._test_i_triples),
            ("kg_test_ii_triples", self._test_ii_triples),
        ):
            if triples:
                payloads[name] = triples_to_graph_payload(triples=triples, **kwargs)

        return payloads
