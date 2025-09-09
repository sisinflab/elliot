"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import copy
from collections import defaultdict
from functools import cached_property
from types import SimpleNamespace

import numpy as np
#import fireducks.pandas as fd
import scipy.sparse as sp
import logging as pylog

#from elliot.dataset.sparse_builder import SparseBuilder
from elliot.negative_sampling.negative_sampling import NegativeSampler
from elliot.utils import logging, sparse


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
        "test_dict",  # comment
    ]

    def __call__(cls, *args, **kwargs):
        class_object = type.__call__(cls, *args, **kwargs)
        cls._check_required_attributes(class_object)
        return class_object

    def _check_required_attributes(cls, class_object):
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
        self.batch_size = 1024
        #self.inverted = {'val_mask': False, 'test_mask': False, 'all_unrated_mask': True}

        self._handle_train_set(side_information_data, data_tuple)
        self._handle_val_test_sets(data_tuple)

        if hasattr(self.config, "negative_sampling"):
            self._handle_negative_sampling()

    def __len__(self):
        return (self.num_users + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        for i, batch_start in enumerate(range(0, self.num_users, self.batch_size)):
            batch_end = batch_start + self.batch_size
            train_data = self.sp_i_train_ratings[batch_start:batch_end]

            neg_val, neg_test = None, None
            if hasattr(self, "val_neg_indices"):
                neg_val = self.val_neg_indices[i]
                #neg_val = (val_mask_portion == 0).nonzero()
            if hasattr(self, "test_neg_indices"):
                neg_test = self.test_neg_indices[i]
                #neg_test = (test_mask_portion == 0).nonzero()
            else:
                neg_test = train_data

            yield train_data, (neg_val, neg_test)

    #def get_batch_mask(self, candidate_negatives, validation=False):
    #    return self.sampler.sample(candidate_negatives, validation) if hasattr(self, "sampler") else None

    def _handle_train_set(self, side_information_data, data_tuple):
        self.train_dict = self._dataframe_to_dict(data_tuple[0])

        if self.config.align_side_with_train == True:
            self.side_information = self._align_with_training(side_information_data=side_information_data)
        else:
            self.side_information = side_information_data

        self.users, self.items = self._get_users_list_and_items_list(self.train_dict)
        self.num_users = len(self.users)
        self.num_items = len(self.items)

        self.transactions = sum(len(v) for v in self.train_dict.values())

        sparsity = 1 - (self.transactions / (self.num_users * self.num_items))
        self.logger.info(
            f"Statistics\tUsers:\t{self.num_users}\tItems:\t{self.num_items}\tTransactions:\t{self.transactions}\t"
            f"Sparsity:\t{sparsity}")

        self.private_users = self.users
        self.public_users = {v: k for k, v in enumerate(self.private_users)}
        self.private_items = self.items
        self.public_items = {v: k for k, v in enumerate(self.private_items)}

        self.i_train_dict = {self.public_users[user]: {self.public_items[i]: v for i, v in items.items()}
                             for user, items in self.train_dict.items()}

        shape = (len(self.users), len(self.items))
        #self.sp_i_train = sparse.build_sparse(self.i_train_dict, shape)
        self.sp_i_train_ratings = sparse.build_sparse_ratings(self.i_train_dict, shape)
        #self.all_unrated_mask = np.where((self.sp_i_train.toarray() == 0), True, False)
        #self.all_rated_mask = self.sp_i_train_ratings.astype(bool)

    #@cached_property
    #def pr_items(self):
    #    return list(self.public_items.values())

    @property
    def sp_i_train(self):
        return sparse.build_sparse(self.i_train_dict, (len(self.users), len(self.items)))

    #@property
    #def sp_i_train_ratings(self):
    #    return sparse.build_sparse_ratings(self.i_train_dict, (len(self.users), len(self.items)))

    #@cached_property
    #def val_mask(self):
    #    return self._val_mask.toarray() if hasattr(self, '_val_mask') else None

    #@cached_property
    #def test_mask(self):
    #    return self._test_mask.toarray() if hasattr(self, '_test_mask') else None

    #@property
    #def all_unrated_mask(self):
        #return np.where((self.sp_i_train_ratings.toarray() == 0), True, False)
    #    self.inverted['all_unrated_mask'] = True
    #    return self.sp_i_train_ratings.astype(bool)

    def _handle_val_test_sets(self, data_tuple):
        if len(data_tuple) == 2:
            self.val_dict = None
            self.test_dict = self._dataframe_to_dict(data_tuple[1])
        else:
            self.val_dict = self._dataframe_to_dict(data_tuple[1])
            self.test_dict = self._dataframe_to_dict(data_tuple[2])

    def _handle_negative_sampling(self):
        sampler = NegativeSampler(self)
        val_neg_indices, test_neg_indices = sampler.sample()
        if val_neg_indices is not None: self.val_neg_indices = val_neg_indices
        if test_neg_indices is not None: self.test_neg_indices = test_neg_indices
        """for is_validation, d, neg_samples, attr in [
            (True, self.val_dict, val_neg_samples, "val_neg_indices"),
            (False, self.test_dict, test_neg_samples, "test_neg_indices")
        ]:
            if is_validation and d is None:
                continue
            sp_matrix = sparse.build_sparse(d, (len(self.public_users), len(self.public_items)),
                                            self.public_users, self.public_items, dtype='bool')
            #self.inverted[attr] = neg_samples
            candidate_items = neg_samples + sp_matrix
            setattr(self, attr, candidate_items)"""

    @staticmethod
    def _dataframe_to_dict(data):
        # users = list(data['userId'].unique())

        "Conversion to Dictionary"
        #ratings = data.set_index('userId')[['itemId', 'rating']].apply(lambda x: (x['itemId'], float(x['rating'])), 1)\
        #    .groupby(level=0).agg(lambda x: dict(x.values)).to_dict()
        #ratings = {
        #    user: {item: float(rating) for item, rating in zip(group['itemId'], group['rating'])}
        #    for user, group in data.groupby('userId')
        #}
        ratings = defaultdict(dict)
        for user, item, rating in zip(data["userId"], data["itemId"], data["rating"]):
            ratings[user][item] = float(rating)
        ratings = dict(ratings)
        # for u in users:
        #     sel_ = data[data['userId'] == u]
        #     ratings[u] = dict(zip(sel_['itemId'], sel_['rating']))
        return ratings

    @staticmethod
    def _get_users_list_and_items_list(dict):
        users = list(dict.keys())
        item_set = set()
        for user_ratings in dict.values():
            item_set.update(user_ratings.keys())
        items = list(item_set)
        return users, items

    """def _build_sparse(self, dict, users, items):
        rows_cols = [(u, i) for u, items in dict.items() for i in items.keys()]
        rows, cols = map(list, zip(*rows_cols))
        #rows, cols = list(rows), list(cols)
        data = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='float32',
                             shape=(len(users), len(items)))
        return data

    def _build_sparse_ratings(self, dict, users, items):
        rows_cols_ratings = [(u, i, r) for u, items in dict.items() for i, r in items.items()]
        rows, cols, ratings = map(list, zip(*rows_cols_ratings))
        #rows = [u for u, _, _ in rows_cols_ratings]
        #cols = [i for _, i, _ in rows_cols_ratings]
        #ratings = [r for _, _, r in rows_cols_ratings]
        data = sp.csr_matrix((ratings, (rows, cols)), dtype='float32',
                             shape=(len(users), len(items)))
        return data

    def _to_bool_sparse(self, test_dict):
        i_test = [(self.public_users[user], self.public_items[i])
                  for user, items in test_dict.items() if user in self.public_users.keys()
                  for i in items.keys() if i in self.public_items.keys()]
        rows, cols = map(list, zip(*i_test))
        #rows = [u for u, _ in i_test]
        #cols = [i for _, i in i_test]
        i_test = sp.csr_matrix((np.ones_like(rows), (rows, cols)), dtype='bool',
                               shape=(len(self.public_users.keys()), len(self.public_items.keys())))
        return i_test"""

    #@property
    #def all_unrated_mask(self):
    #    return self.sp_i_train.toarray() == 0

    #@cached_property
    #def sp_i_train_ratings(self):
    #    return SparseBuilder.build_sparse_ratings(self.i_train_dict, self.users, self.items)

    def get_test(self):
        return self.test_dict

    def get_validation(self):
        return self.val_dict if hasattr(self, 'val_dict') else None

    def build_items_neighbour(self):
        row, col = self.sp_i_train.nonzero()
        edge_index = np.array([row, col])
        iu_dict = {i: edge_index[0, iu].tolist() for i, iu in
                   enumerate(list((edge_index[1] == i).nonzero()[0] for i in list(self.private_items.keys())))}
        return iu_dict

    def _align_with_training(self, side_information_data):
        """Alignment with training"""

        def equal(a, b, c):
            return len(a) == len(b) == len(c)

        users = set(self.train_dict.keys())
        items = set({k for a in self.train_dict.values() for k in a.keys()})
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
