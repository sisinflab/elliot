from collections import Counter
from types import SimpleNamespace
import pandas as pd
import typing as t

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class ChainedKG(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.attribute_file = getattr(ns, "map", None)
        self.feature_file = getattr(ns, "features", None)
        self.properties_file = getattr(ns, "properties", None)
        self.additive = getattr(ns, "additive", True)
        self.threshold = getattr(ns, "threshold", 10)
        self.users = users
        self.items = items
        
        if (self.attribute_file is not None) & (self.feature_file is not None) & (self.properties_file is not None):
            self.map_ = self.load_attribute_file(self.attribute_file)
            self.feature_names = self.load_feature_names(self.feature_file)
            self.properties = self.load_properties(self.properties_file)
            self.map_ = self.reduce_attribute_map_property_selection(self.map_, self.items, self.feature_names, self.properties, self.additive, self.threshold)
            self.items = self.items & set(self.map_.keys())

    def get_mapped(self):
        return self.users, self.items

    def filter(self, users, items):
        self.users = self.users & users
        self.items = self.items & items
        self.map_ = {k: v for k, v in self.map_.items() if k in self.items}

        self.map_ = self.reduce_attribute_map_property_selection(self.map_, self.items, self.feature_names,
                                                                 self.properties, self.additive, self.threshold)
        self.items = self.items & set(self.map_.keys())

    def create_namespace(self):
        ns = SimpleNamespace()
        ns.__name__ = "ChainedKG"
        ns.object = self
        ns.feature_map = self.map_
        ns.features = list({f for i in self.items for f in ns.feature_map[i]})
        ns.nfeatures = len(ns.features)
        ns.private_features = {p: f for p, f in enumerate(ns.features)}
        ns.public_features = {v: k for k, v in ns.private_features.items()}
        return ns

    def load_attribute_file(self, attribute_file, separator='\t'):
        map = {}
        with open(attribute_file) as file:
            for line in file:
                line = line.split(separator)
                int_list = [int(i) for i in line[1:]]
                map[int(line[0])] = list(set(int_list))
        return map

    def load_item_set(self, ratings_file, separator='\t', itemPosition=1):
        s = set()
        with open(ratings_file) as file:
            for line in file:
                line = line.split(separator)
                s.add(int(line[itemPosition]))
        return s

    def load_feature_names(self, infile, separator='\t'):
        feature_names = {}
        with open(infile) as file:
            for line in file:
                line = line.split(separator)
                pattern = line[1].split('><')
                pattern[0] = pattern[0][1:]
                pattern[len(pattern) - 1] = pattern[len(pattern) - 1][:-2]
                feature_names[int(line[0])] = pattern
        return feature_names

    def load_properties(self, properties_file):
        properties = []
        with open(properties_file) as file:
            for line in file:
                if line[0] != '#':
                    properties.append(line.rstrip("\n"))
        return properties

    def reduce_attribute_map_property_selection(self, map, items, feature_names, properties, additive, threshold = 10):

        acceptable_features = set()
        if not properties:
            acceptable_features.update(feature_names.keys())
        else:
            for feature in feature_names.items():
                if additive:
                    if feature[1][0] in properties:
                        acceptable_features.add(int(feature[0]))
                else:
                    if feature[1][0] not in properties:
                        acceptable_features.add(int(feature[0]))

        self.logger.info(f"Acceptable Features:\t{len(acceptable_features)}\tMapped items:\t{len(map)}")

        nmap = {k: v for k, v in map.items() if k in items}

        feature_occurrences_dict = Counter([x for xs in nmap.values() for x in xs  if x in acceptable_features])
        features_popularity = {k: v for k, v in feature_occurrences_dict.items() if v > threshold}

        self.logger.info(f"Features above threshold:\t{len(features_popularity)}")

        new_map = {k:[value for value in v if value in features_popularity.keys()] for k,v in nmap.items()}
        new_map = {k:v for k,v in new_map.items() if len(v)>0}
        self.logger.info(f"Final #items:\t{len(new_map.keys())}")

        return new_map
