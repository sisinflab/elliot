"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import pandas as pd
import scipy.sparse as sp

from dataset.abstract_dataset import AbstractDataset


class DataSet(AbstractDataset):
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
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating']
        self.train_dataframe = pd.read_csv(path_train_data, sep="\t", header=None, names=self.column_names)
        self.users = list(self.train_dataframe['userId'].unique())
        self.train_dict = self.build_dict(self.train_dataframe, self.users)
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_users = len(self.users)
        self.num_items = self.train_dataframe['itemId'].nunique()
        self.transactions = len(self.train_dataframe)

        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}

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

    # def build_sparse(self, dataframe_dict, num_users, num_items):
    #     train = sp.dok_matrix((num_users, num_items), dtype=np.float32)
    #     for user, user_items in dataframe_dict.items():
    #         for item, rating in user_items.items():
    #             if rating > 0:
    #                 train[user, item] = 1.0
    #     return train
    #
    # #non capisco bene cosa faccia
    # def load_train_file_as_list(self, filename):
    #     # Get number of users and items
    #     u_ = 0
    #     self.train_list, items = [], []
    #     with open(filename, "r") as f:
    #         line = f.readline()
    #         index = 0
    #         while line is not None and line != "":
    #             arr = line.split("\t")
    #             u, i = int(arr[0]), int(arr[1])
    #             if u_ < u:
    #                 index = 0
    #                 self.train_list.append(items)
    #                 items = []
    #                 u_ += 1
    #             index += 1
    #             items.append(i)
    #             line = f.readline()
    #     self.train_list.append(items)

    # def prepare_test(self, dataframe):
    #     return dataframe[["userId","itemId"]].values

    def get_test(self):
        return self.test_dict
