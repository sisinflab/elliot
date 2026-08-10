from typing import Dict, Set, Any
import numpy as np
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.formats import GraphPayload
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="kg_edges",
    format="graph"
)
class KGINTSVLoader(AbstractLoader):
    """KG triples for KGIN-style relation-aware propagation, read from a headerless
    tabular file of integer `(head, relation, tail)` triples via `Reader.read_tabular`
    -- set `reader.sep` to a single space in the side-info config for whitespace-
    separated dumps. Always fully loads into memory: building `entity2id`/`relation2id`
    needs every triple read and indexed up front regardless.
    """

    attribute_file: str

    def __init__(self, **params):
        super().__init__(**params)

        # Initializing variables
        self._map: pd.DataFrame = pd.DataFrame()
        self._entities: Set[Any] = set()
        self._entity_list: Set[Any] = set()

        self._map = self.reader.read_tabular(
            path=self.attribute_file,
            header=self._reader_config.header,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding
        )

        self._entities = set(self._map.values[:, 0]) | set(self._map.values[:, 2])
        self.items = self.items & self._entities
        self._entity_list = self._entities - self.items

    def filter(self, users, items):
        super().filter(users, items)
        self._map = self._map[self._map[self._map.columns[0]].isin(self.items)]

        self._entities = set(self._map.values[:, 0]) | set(self._map.values[:, 2])
        self.items = self.items & self._entities
        self._entity_list = self._entities - self.items

    def load(self) -> Dict[str, GraphPayload]:
        """Return the KG triples in the canonical `GraphPayload` encoding: parallel
        (head, relation, tail) int-id arrays plus an `entity2id`/`relation2id` index,
        normalizing this loader's pandas-DataFrame-of-raw-ids representation: items are
        indexed first, then the remaining non-item entities (offset by `len(items)`).
        """
        heads_raw = self._map.values[:, 0]
        relations_raw = self._map.values[:, 1]
        tails_raw = self._map.values[:, 2]

        items_sorted = sorted(self.items)
        entity_list_sorted = sorted(self._entity_list)
        entity2id = {e: i for i, e in enumerate(items_sorted)}
        entity2id.update({e: i + len(items_sorted) for i, e in enumerate(entity_list_sorted)})

        relations_sorted = sorted(set(relations_raw.tolist()))
        # Offset by 1, reserving relation id 0.
        relation2id = {r: i + 1 for i, r in enumerate(relations_sorted)}

        heads = np.array([entity2id[h] for h in heads_raw], dtype=np.int64)
        tails = np.array([entity2id[t] for t in tails_raw], dtype=np.int64)
        relations = np.array([relation2id[r] for r in relations_raw], dtype=np.int64)

        item_entity_map = {item: entity2id[item] for item in items_sorted}

        payload = GraphPayload(
            heads=heads,
            relations=relations,
            tails=tails,
            entity2id=entity2id,
            relation2id=relation2id,
            item_entity_map=item_entity_map,
        )
        return {"kg_triples": payload}
