from types import SimpleNamespace
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class ItemAttributes(AbstractLoader):
    def __init__(self, data: pd.DataFrame, ns: SimpleNamespace):
        self.data = data.copy(deep=True)
        self.attribute_file = getattr(ns, "attribute_file", None)

        self.map_ = self.load_attribute_file(self.attribute_file)
        self.items = set(self.map_.keys())
        self.data = self.data[self.data['itemId'].isin(self.items)]
        self.users = set(self.data['userId'].unique())
        self.items = set(self.data['itemId'].unique())

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.data = self.data[self.data['userId'].isin(users)]
        self.data = self.data[self.data['itemId'].isin(items)]
        self.users = set(self.data['userId'].unique())
        self.items = set(self.data['itemId'].unique())

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "ItemAttributes"
        ns.feature_map = self.map_
        ns.features = list({f for i in self.items for f in ns.feature_map[i]})
        ns.nfeatures = len(ns.features)
        ns.private_features = {p: f for p, f in enumerate(ns.features)}
        ns.public_features = {v: k for k, v in ns.private_features.items()}
        return ns

    def load_attribute_file(self, attribute_file, separator='\t'):
        map_ = {}
        with open(attribute_file) as file:
            for line in file:
                line = line.split(separator)
                int_list = [int(i) for i in line[1:]]
                map_[int(line[0])] = list(set(int_list))
        return map_
