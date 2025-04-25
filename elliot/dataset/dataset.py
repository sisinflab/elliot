"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import copy
from types import SimpleNamespace

import numpy as np
#import fireducks.pandas as fd
import scipy.sparse as sp
import logging as pylog

from elliot.negative_sampling.negative_sampling import NegativeSampler
from elliot.utils import logging


class DataSetRequiredAttributesController(type):
    required_attributes = [
        "config",  # comment
        "args",  # comment
        "kwargs",  # comment
        "users",  # comment
        "items",  # comment
        "num_users",  # comment
        "num_items",  # comment
        "private_users",  # comment
        "public_users",  # comment
        "private_items",  # comment
        "public_items",  # comment
        "transactions",  # comment
        "train_dict",  # comment
        "i_train_dict",  # comment
        "sp_i_train",  # comment
        "test_dict"  # comment
    ]

    def __call__(cls, *args, **kwargs):
        class_object = type.__call__(cls, *args, **kwargs)
        cls.check_required_attributes(class_object)
        return class_object

    def check_required_attributes(cls, class_object):
        missing_attrs = [f"{attr}" for attr in cls.required_attributes
                         if not hasattr(class_object, attr)]
        if missing_attrs:
            raise NotImplementedError("class '%s' requires attribute%s %s" %
                                      (class_object.__class__.__name__, "s" * (len(missing_attrs) > 1),
                                       ", ".join(missing_attrs)))


class DataSet(metaclass=DataSetRequiredAttributesController):
    """
    Load train and test dataset
    """

    def __init__(self, config, data_tuple, side_information_data, *args, **kwargs):
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """
        self.logger = logging.get_logger(self.__class__.__name__, pylog.CRITICAL if config.config_test else
        pylog.DEBUG)
        self.config = config
        self.args = args
        self.kwargs = kwargs

        if self.config.align_side_with_train == True:
            self.side_information = self.align_with_training(train=data_tuple[0],
                                                             side_information_data=side_information_data)
        else:
            self.side_information = side_information_data

        self.train_dict = self.dataframe_to_dict(data_tuple[0])

        self.users = list(self.train_dict.keys())
        self.items = list({k for a in self.train_dict.values() for k in a.keys()})
        self.num_users = len(self.users)
        self.num_items = len(self.items)
        self.transactions = sum(len(v) for v in self.train_dict.values())

        sparsity = 1 - (self.transactions / (self.num_users * self.num_items))
        self.logger.info(
            f"Statistics\tUsers:\t{self.num_users}\tItems:\t{self.num_items}\tTransactions:\t{self.transactions}\t"
            f"Sparsity:\t{sparsity}")

        self.private_users = {p: u for p, u in enumerate(self.users)}
        self.public_users = {v: k for k, v in self.private_users.items()}
        self.private_items = {p: i for p, i in enumerate(self.items)}
        self.public_items = {v: k for k, v in self.private_items.items()}

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                             for user, items in self.train_dict.items()}

        self.sp_i_train = self.build_sparse()
        self.sp_i_train_ratings = self.build_sparse_ratings()

        if len(data_tuple) == 2:
            self.test_dict = self.build_dict(data_tuple[1])
            if hasattr(config, "negative_sampling"):
                val_neg_samples, test_neg_samples = NegativeSampler.sample(config, self.public_users, self.public_items,
                                                                           self.private_users, self.private_items,
                                                                           self.sp_i_train, None, self.test_dict)
                sp_i_test = self.to_bool_sparse(self.test_dict)
                test_candidate_items = test_neg_samples + sp_i_test
                self.test_mask = np.where((test_candidate_items.toarray() == True), True, False)
        else:
            self.val_dict = self.build_dict(data_tuple[1])
            self.test_dict = self.build_dict(data_tuple[2])
            if hasattr(config, "negative_sampling"):
                val_neg_samples, test_neg_samples = NegativeSampler.sample(config, self.public_users, self.public_items,
                                                                           self.private_users, self.private_items,
                                                                           self.sp_i_train, self.val_dict,
                                                                           self.test_dict)
                sp_i_val = self.to_bool_sparse(self.val_dict)
                sp_i_test = self.to_bool_sparse(self.test_dict)
                val_candidate_items = val_neg_samples + sp_i_val
                self.val_mask = np.where((val_candidate_items.toarray() == True), True, False)
                test_candidate_items = test_neg_samples + sp_i_test
                self.test_mask = np.where((test_candidate_items.toarray() == True), True, False)

        self.allunrated_mask = np.where((self.sp_i_train.toarray() == 0), True, False)

    def build_items_neighbour(self):
        row, col = self.sp_i_train.nonzero()
        edge_index = np.array([row, col])
        iu_dict = {i: edge_index[0, iu].tolist() for i, iu in
                   enumerate(list((edge_index[1] == i).nonzero()[0] for i in list(self.private_items.keys())))}
        return iu_dict

    def dataframe_to_dict(self, data):
        # users = list(data['userId'].unique())

        "Conversion to Dictionary"
        #ratings = data.set_index('userId')[['itemId', 'rating']].apply(lambda x: (x['itemId'], float(x['rating'])), 1)\
        #    .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
        ratings = {
            user: {item: float(rating) for item, rating in zip(group['itemId'], group['rating'])}
            for user, group in data.groupby('userId')
        }

        # for u in users:
        #     sel_ = data[data['userId'] == u]
        #     ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings

    def build_dict(self, dataframe):
        #ratings = dataframe.set_index('userId')[['itemId', 'rating']].apply(lambda x: (x['itemId'], float(x['rating'])), 1)\
        #    .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
        ratings = {
            user: {item: float(rating) for item, rating in zip(group['itemId'], group['rating'])}
            for user, group in dataframe.groupby('userId')
        }
        # for u in users:
        #     sel_ = dataframe[dataframe['userId'] == u]
        #     ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
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

    def to_bool_sparse(self, test_dict):
        i_test = [(self.public_users[user], self.public_items[i])
                  for user, items in test_dict.items() if user in self.public_users.keys()
                  for i in items.keys() if i in self.public_items.keys()]
        rows = [u for u, _ in i_test]
        cols = [i for _, i in i_test]
        i_test = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='bool',
                               shape=(len(self.public_users.keys()), len(self.public_items.keys())))
        return i_test

    def align_with_training(self, train, side_information_data):
        """Alignment with training"""

        def equal(a, b, c):
            return len(a) == len(b) == len(c)

        train_dict = self.dataframe_to_dict(train)
        users = set(train_dict.keys())
        items = set({k for a in train_dict.values() for k in a.keys()})
        users_items = []
        side_objs = []
        for k, v in side_information_data.__dict__.items():
            new_v = copy.deepcopy(v)
            users_items.append(new_v.object.get_mapped())
            side_objs.append(new_v)
        while True:
            condition = True
            new_users = users
            new_items = items
            for us_, is_ in users_items:
                temp_users = new_users & us_
                temp_items = new_items & is_
                condition &= equal(new_users, us_, temp_users)
                condition &= equal(new_items, is_, temp_items)
                new_users = temp_users
                new_items = temp_items
            if condition:
                break
            else:
                users = new_users
                items = new_items
                new_users_items = []
                for v in side_objs:
                    v.object.filter(users, items)
                    new_users_items.append(v.object.get_mapped())
                users_items = new_users_items
        ns = SimpleNamespace()
        for side_obj in side_objs:
            side_ns = side_obj.object.create_namespace()
            name = side_ns.__name__
            setattr(ns, name, side_ns)
        return ns
