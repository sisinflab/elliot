from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.utils.enums import AlignmentMode
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="sparse",
    alignment=AlignmentMode.DROP
)
class ItemAttributes(AbstractLoader):
    attribute_file: str

    def __init__(self, **params):
        super().__init__(**params)

        self.map_ = self.load_attribute_file()

        self.items = self.items & set(self.map_.keys())
        self.logger.debug(
            "Loaded item attributes",
            extra={
                "context": {
                    "source": self.name,
                    "items_with_features": len(self.items),
                    "unique_features": len(set(f for feats in self.map_.values() for f in feats)),
                }
            },
        )

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = self.name
        ns.object = self
        ns.feature_map = self.map_
        ns.features = list({f for i in self.items for f in ns.feature_map[i]})
        ns.nfeatures = len(ns.features)
        ns.private_features = {p: f for p, f in enumerate(ns.features)}
        ns.public_features = {v: k for k, v in ns.private_features.items()}
        return ns

    def load_attribute_file(self):
        map_ = self.reader.read_mapping(
            path=self.attribute_file,
            sep=self._reader_config.sep,
            dtype="int"
        )
        return map_
