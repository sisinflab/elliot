"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd
import scipy.sparse as sp
from collections import Counter

from dataset.abstract_dataset import AbstractDataset


class KnowledgeChains(AbstractDataset):
    """
    Load train and test dataset
    """

    def __init__(self, config, *args, **kwargs):
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """

        path_train_data = config.data_paths.train_data
        path_test_data = config.data_paths.test_data
        path_map = config.data_paths.map
        path_features = config.data_paths.features
        path_properties = config.data_paths.properties

        self.config = config
        self.column_names = ['userId', 'itemId', 'rating']

        self.train_dict, self.feature_map = self.load_dataset_dict(path_train_data,
                                                                             "\t",
                                                                             path_map,
                                                                             path_features,
                                                                             path_properties)

        self.users = list(self.train_dict.keys())
        self.num_users = len(self.users)
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_items = len(self.items)

        self.features = list({f for i in self.items for f in self.feature_map[i]})
        self.factors = len(self.features)
        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}
        self.private_features = {p: f for p, f in enumerate(self.features)}
        self.public_features = {v: k for k, v in self.private_features.items()}
        self.transactions = sum(len(v) for v in self.train_dict.values())

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                                for user, items in self.train_dict.items()}

        self.sp_i_train = self.build_sparse()

        self.test_dataframe = pd.read_csv(path_test_data, sep="\t", header=None, names=self.column_names)
        self.test_dict = self.build_dict(self.test_dataframe, self.users)

        print('{0} - Loaded'.format(path_train_data))
        # self.params = params
        self.args = args
        self.kwargs = kwargs

    def build_dict(self, dataframe, users):
        ratings = {}
        for u in users:
            sel_ = dataframe[dataframe['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings

    def build_sparse(self):

        rows_cols = [(u, i) for u, items in self.i_train_dict.items() for i in items.keys()]
        rows = [u for u, _ in rows_cols]
        cols = [i for _, i in rows_cols]
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))
        return data

    def get_test(self):
        return self.test_dict

    def load_dataset_dict(self, file_ratings, separator='\t', attribute_file=None, feature_file=None, properties_file=None,
                          additive=True, threshold=10):
        column_names = ['userId', 'itemId', 'rating']
        data = pd.read_csv(file_ratings, sep=separator, header=None, names=column_names)
        if (attribute_file is not None) & (feature_file is not None) & (properties_file is not None):
            map = self.load_attribute_file(attribute_file)
            items = self.load_item_set(file_ratings)
            feature_names = self.load_feature_names(feature_file)
            properties = self.load_properties(properties_file)
            map = self.reduce_attribute_map_property_selection(map, items, feature_names, properties, additive, threshold)
            items = set(map.keys())
            data = data[data[column_names[1]].isin(items)]
        users = list(data['userId'].unique())

        "Conversion to Dictionary"
        ratings = {}
        for u in users:
            sel_ = data[data['userId'] == u]
            ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        n_users = len(ratings.keys())
        n_items = len({k for a in ratings.values() for k in a.keys()})
        transactions = sum([len(a) for a in ratings.values()])
        sparsity = 1 - (transactions / (n_users * n_items))
        print()
        print("********** Statistics")
        print(f'Users:\t{n_users}')
        print(f'Items:\t{n_items}', )
        print(f'Transactions:\t{transactions}')
        print(f'Sparsity:\t{sparsity}')
        print("********** ")
        return ratings, map

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

        print(f"Acceptable Features:\t{len(acceptable_features)}", )
        print(f"Mapped items:\t{len(map)}")

        nmap = {k: v for k, v in map.items() if k in items}

        feature_occurrences_dict = Counter([x for xs in nmap.values() for x in xs  if x in acceptable_features])
        features_popularity = {k: v for k, v in feature_occurrences_dict.items() if v > threshold}

        print(f"Features above threshold:\t{len(features_popularity)}")

        new_map = {k:[value for value in v if value in features_popularity.keys()] for k,v in nmap.items()}
        new_map = {k:v for k,v in new_map.items() if len(v)>0}
        print(f"Final #items:\t{len(new_map.keys())}")

        return new_map

    def reduce_dataset_by_item_list(self, ratings_file, items, separator = '\t'):
        column_names = ["userId", "itemId", "rating"]
        data = pd.read_csv(ratings_file, sep=separator, header=None, names=column_names)
        data = data[data[column_names[1]].isin(items)]
        return data
