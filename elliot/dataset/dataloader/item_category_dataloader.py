"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import concurrent.futures as c
import logging as pylog
import os
import typing as t
from ast import literal_eval
from types import SimpleNamespace

import numpy as np
import pandas as pd
import scipy.sparse as sp

from elliot.prefiltering.standard_prefilters import PreFilter
from elliot.splitter.base_splitter import Splitter
from elliot.utils import logging

"""
[(train_0,test_0)]
[([(train_0,val_0)],test_0)]
[data_0]

[([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_0),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_1),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_2),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_3),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_4)]

[[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4],
[data_0,data_1,data_2,data_3,data_4]]

[[data_0],[data_1],[data_2]]

[[data_0,data_1,data_2]]

[[data_0,data_1,data_2],[data_0,data_1,data_2],[data_0,data_1,data_2]]
"""


class ItemCategoryLoader:
    """
    Load train and test dataset
    """

    def __init__(self, config, *args, **kwargs):
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """

        self.logger = logging.get_logger(self.__class__.__name__)
        self.args = args
        self.kwargs = kwargs
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating', 'timestamp']
        if config.config_test:
            return

        self.side_information_data = SimpleNamespace()

        if config.data_config.strategy == "fixed":
            path_train_data = config.data_config.train_path
            path_val_data = getattr(config.data_config, "validation_path", None)
            path_test_data = config.data_config.test_path
            item_mapping_path = getattr(config.data_config.side_information, "item_mapping", None)


            self.train_dataframe, self.side_information_data.feature_map = self.load_dataset_dataframe(path_train_data,
                                                                                                 "\t",
                                                                                                 item_mapping_path,
                                                                                                 self.column_names)
            self.side_information_data.item_mapping_path = item_mapping_path

            self.train_dataframe = self.check_timestamp(self.train_dataframe)

            self.logger.info('{0} - Loaded'.format(path_train_data))

            self.test_dataframe = pd.read_csv(path_test_data, sep="\t", header=None, names=self.column_names)
            self.test_dataframe = self.check_timestamp(self.test_dataframe)

            if path_val_data:
                self.validation_dataframe = pd.read_csv(path_val_data, sep="\t", header=None, names=self.column_names)
                self.validation_dataframe = self.check_timestamp(self.validation_dataframe)

                self.tuple_list = [([(self.train_dataframe, self.validation_dataframe)], self.test_dataframe)]
            else:
                self.tuple_list = [(self.train_dataframe, self.test_dataframe)]

        elif config.data_config.strategy == "hierarchy":
            item_mapping_path = getattr(config.data_config.side_information, "item_mapping", None)
            self.side_information_data.feature_map = self.load_attribute_file(item_mapping_path)

            self.tuple_list = self.read_splitting(config.data_config.root_folder)

            self.logger.info('{0} - Loaded'.format(config.data_config.root_folder))

        elif config.data_config.strategy == "dataset":
            self.logger.info("There will be the splitting")
            path_dataset = config.data_config.dataset_path

            item_mapping_path = getattr(config.data_config.side_information, "item_mapping", None)

            self.dataframe, self.side_information_data.feature_map = self.load_dataset_dataframe(path_dataset,
                                                                                                 "\t",
                                                                                                 item_mapping_path,
                                                                                                 self.column_names)
            self.side_information_data.item_mapping_path = item_mapping_path

            self.dataframe = self.check_timestamp(self.dataframe)

            self.logger.info('{0} - Loaded'.format(path_dataset))

            self.dataframe = PreFilter.filter(self.dataframe, self.config)

            splitter = Splitter(self.dataframe, self.config.splitting)
            self.tuple_list = splitter.process_splitting()

        else:
            raise Exception("Strategy option not recognized")

    def check_timestamp(self, d: pd.DataFrame) -> pd.DataFrame:
        if all(d["timestamp"].isna()):
            d = d.drop(columns=["timestamp"]).reset_index(drop=True)
        return d

    def read_splitting(self, folder_path):
        tuple_list = []
        for dirs in os.listdir(folder_path):
            for test_dir in dirs:
                test_ = pd.read_csv(f"{folder_path}{test_dir}/test.tsv", sep="\t")
                val_dirs = [f"{folder_path}{test_dir}/{val_dir}/" for val_dir in os.listdir(f"{folder_path}{test_dir}") if os.path.isdir(f"{folder_path}{test_dir}/{val_dir}")]
                val_list = []
                for val_dir in val_dirs:
                    train_ = pd.read_csv(f"{val_dir}/train.tsv", sep="\t")
                    val_ = pd.read_csv(f"{val_dir}/val.tsv", sep="\t")
                    val_list.append((train_, val_))
                if not val_list:
                    val_list = pd.read_csv(f"{folder_path}{test_dir}/train.tsv", sep="\t")
                tuple_list.append((val_list, test_))

        return tuple_list

    def generate_dataobjects(self) -> t.List[object]:
        data_list = []
        for train_val, test in self.tuple_list:
            # testset level
            if isinstance(train_val, list):
                # validation level
                val_list = []
                for train, val in train_val:
                    single_dataobject = ItemCategoryDataObject(self.config, (train,val,test), self.side_information_data, self.args, self.kwargs)
                    val_list.append(single_dataobject)
                data_list.append(val_list)
            else:
                single_dataobject = ItemCategoryDataObject(self.config, (train_val, test), self.side_information_data, self.args,
                                                     self.kwargs)
                data_list.append([single_dataobject])
        return data_list

    def generate_dataobjects_mock(self) -> t.List[object]:
        _column_names = ['userId', 'itemId', 'rating']
        training_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))
        test_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))

        side_information_data = SimpleNamespace()

        training_set = pd.DataFrame(np.array(training_set), columns=_column_names)
        test_set = pd.DataFrame(np.array(test_set), columns=_column_names)

        side_information_data.feature_map = {item: np.random.randint(0, 10, size=np.random.randint(0, 20)).tolist()
                                             for item in training_set['itemId'].unique()}

        data_list = [[ItemCategoryDataObject(self.config, (training_set, test_set), side_information_data,
                                                self.args, self.kwargs)]]

        return data_list

    def load_dataset_dataframe(self, file_ratings,
                               separator='\t',
                               attribute_file=None,
                               column_names=['userId', 'itemId', 'rating', 'timestamp']
                               ):
        data = pd.read_csv(file_ratings, sep=separator, header=None, names=column_names)
        if (attribute_file is not None):
            map = self.load_attribute_file(attribute_file)
            items = set(data['itemId'].unique().tolist())
            map = {k: v for k, v in map.items() if k in items}
            items = set(map.keys())
            data = data[data[column_names[1]].isin(items)]

        return data, map

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

    def reduce_dataset_by_item_list(self, ratings_file, items, separator = '\t'):
        column_names = ["userId", "itemId", "rating"]
        data = pd.read_csv(ratings_file, sep=separator, header=None, names=column_names)
        data = data[data[column_names[1]].isin(items)]
        return data


class ItemCategoryDataObject:
    """
    Load train and test dataset
    """

    def __init__(self, config, data_tuple, side_information_data, *args, **kwargs):
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG)
        self.config = config
        self.side_information_data = side_information_data
        self.args = args
        self.kwargs = kwargs
        self.train_dict = self.dataframe_to_dict(data_tuple[0])

        self.users = list(self.train_dict.keys())
        self.num_users = len(self.users)
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_items = len(self.items)

        self.features = list({f for i in self.items for f in self.side_information_data.feature_map[i]})
        self.nfeatures = len(self.features)
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
        self.sp_i_train_ratings = self.build_sparse_ratings()

        if len(data_tuple) == 2:
            self.test_dict = self.build_dict(data_tuple[1], self.users)
        else:
            self.val_dict = self.build_dict(data_tuple[1], self.users)
            self.test_dict = self.build_dict(data_tuple[2], self.users)

        self.allunrated_mask = np.where((self.sp_i_train.toarray() == 0), True, False)

    def dataframe_to_dict(self, data):
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
        self.logger.info(f"Statistics\tUsers:\t{n_users}\tItems:\t{n_items}\tTransactions:\t{transactions}\t"
                         f"Sparsity:\t{sparsity}")
        return ratings

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

    def build_sparse_ratings(self):
        rows_cols_ratings = [(u, i, r) for u, items in self.i_train_dict.items() for i, r in items.items()]
        rows = [u for u, _, _ in rows_cols_ratings]
        cols = [i for _, i, _ in rows_cols_ratings]
        ratings = [r for _, _, r in rows_cols_ratings]

        data = sp.csr_matrix((ratings, (rows, cols)), dtype='float32',
                             shape=(len(self.users), len(self.items)))

        return data

    def get_test(self):
        return self.test_dict

    def get_validation(self):
        return self.val_dict if hasattr(self, 'val_dict') else None


