"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from types import SimpleNamespace
from scipy.sparse import csr_matrix

import copy
import numpy as np
import logging as pylog
from collections import defaultdict
from functools import cached_property
from tqdm import tqdm

import torch
from torch.utils.data import TensorDataset, Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch_sparse import SparseTensor

from elliot.negative_sampling import NegativeSampler
from elliot.utils import logging


class NegEvalDataset(Dataset):
    def __init__(self, dataset_obj):
        self.num_users = dataset_obj.num_users

        pos, val, test = dataset_obj.get_positive_items()
        self.pos_items = pos
        self.val_items = val
        self.test_items = test

        sampler = NegativeSampler(
            namespace=dataset_obj.config.negative_sampling,
            mappings=dataset_obj.get_mappings(),
            inv_mappings=dataset_obj.get_inverse_mappings(),
            pos_items=self.pos_items,
            add_validation_sampling=len(self.val_items) > 0
        )
        val_neg_items, test_neg_items = sampler.sample()

        self.val_neg_items = self._add_indices(val_neg_items, validation=True)
        self.test_neg_items = self._add_indices(test_neg_items)

    def _add_indices(self, neg, validation=False):
        """Add test or validation samples to the sampled negatives."""
        if neg is None:
            return None

        total = len(neg)
        additional_items = self.val_items if validation else self.test_items
        final_items = []
        i = 0
        text = "validation" if validation else "test"

        with tqdm(total=total, desc=f"Adding {text} items to sampled negatives", leave=False) as t:
            while i < len(neg):
                a, b = neg[i], additional_items[i]
                final_items.append(torch.tensor(a + b))

                # Manual garbage collection
                del neg[i]
                del additional_items[i]

                t.update()

        return final_items

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        test_neg_items = self.test_neg_items[idx]
        val_neg_items = None
        if self.val_neg_items is not None:
            val_neg_items = self.val_neg_items[idx]
        return idx, val_neg_items, test_neg_items


class NegEvalDataLoader(DataLoader):
    def __init__(self, neg_eval_dataset, batch_size):
        super().__init__(
            neg_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(batch):
        user_indices, val_negatives, test_negatives = zip(*batch)

        # User indices will be a list of ints, so we convert it
        user_indices = torch.tensor(list(user_indices))

        # We use the pad_sequence utility to pad item indices
        # in order to have all tensor of the same size
        test_neg_padded = pad_sequence(
            test_negatives,
            batch_first=True,
            padding_value=-1,
        )
        val_neg_padded = None

        if all(v is not None for v in val_negatives):
            val_neg_padded = pad_sequence(
                val_negatives,
                batch_first=True,
                padding_value=-1,
            )

        return user_indices, val_neg_padded, test_neg_padded


class FullEvalDataset(Dataset):
    def __init__(self, dataset_obj):
        self.num_users = dataset_obj.num_users

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return idx


class FullEvalDataloader(DataLoader):
    def __init__(self, full_eval_dataset, batch_size):
        super().__init__(
            full_eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )

    @staticmethod
    def _collate_fn(batch):
        return torch.tensor(batch)


class DataSet:
    """
    Load train and test dataset
    """

    def __init__(self, config, data_tuple, side_information_data, *args, **kwargs):
        """
        Constructor of DataSet
        :param path_train_data: relative path for train file
        :param path_test_data: relative path for test file
        """
        self.logger = logging.get_logger(
            self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG
        )
        self.config = config
        self.args = args
        self.kwargs = kwargs
        self.batch_size = 1024
        self.cold_items = set()
        self.cold_users = set()

        self._handle_train_set(side_information_data, data_tuple)
        self._handle_val_test_sets(data_tuple)

        if hasattr(self.config, "negative_sampling"):
            self.neg_eval_dataset = NegEvalDataset(self)
            self.full_eval_dataset = None
        else:
            self.neg_eval_dataset = None
            self.full_eval_dataset = FullEvalDataset(self)

        self._cached_samplers = {}

    def _handle_train_set(self, side_information_data, data_tuple):
        self._train_dict = self._dataframe_to_dict(data_tuple[0])

        if self.config.align_side_with_train:
            self.side_information = self._align_with_training(side_information_data)
        else:
            self.side_information = side_information_data

        self.users, self.items = self._get_users_and_items()
        self.num_users = len(self.users)
        self.num_items = len(self.items)

        self.transactions = sum(len(v) for v in self._train_dict.values())

        sparsity = 1 - (self.transactions / (self.num_users * self.num_items))
        self.logger.info(
            f"Statistics\t"
            f"Users:\t{self.num_users}\t"
            f"Items:\t{self.num_items}\t"
            f"Transactions:\t{self.transactions}\t"
            f"Sparsity:\t{sparsity}"
        )

        self._u_map = {user: k for k, user in enumerate(self.users)}
        self._i_map = {item: k for k, item in enumerate(self.items)}

        self._i_train_dict = self._build_mapped_dict(self._train_dict)

        rows, cols, data = self._get_triples()
        self.sp_i_train_ratings = csr_matrix(
            (data, (rows, cols)), dtype=float, shape=(self.num_users, self.num_items)
        )

        if self.side_information:
            self._annotate_side_information()
            self._log_side_information()
            self.fuser = FeatureFuser(self.side_information)
        else:
            self.side_information = None
            self.fuser = None

    def _handle_val_test_sets(self, data_tuple):
        if len(data_tuple) == 2:
            self._val_dict = None
            self._test_dict = self._dataframe_to_dict(data_tuple[1])
        else:
            self._val_dict = self._dataframe_to_dict(data_tuple[1])
            self._test_dict = self._dataframe_to_dict(data_tuple[2])

        self._i_val_dict = self._build_mapped_dict(self._val_dict)
        self._i_test_dict = self._build_mapped_dict(self._test_dict)

    def _dataframe_to_dict(self, data, skip_cold_users_items=True):
        """Conversion to Dictionary"""
        ratings_dict = defaultdict(dict)
        users, items, ratings = data["userId"], data["itemId"], data["rating"]

        iter_df = tqdm(
            zip(users, items, ratings),
            total=len(users),
            desc=f"Building ratings dict",
            leave=False
        )

        u_map = getattr(self, "_u_map", None)
        i_map = getattr(self, "_i_map", None)

        for user, item, rating in iter_df:
            if skip_cold_users_items:
                # Cold user?
                if u_map is not None and user not in u_map:
                    self.cold_users.add(user)
                    # And cold item?
                    if i_map is not None and item not in i_map:
                        self.cold_items.add(item)
                    continue

                # Cold item?
                if i_map is not None and item not in i_map:
                    self.cold_items.add(item)
                    continue

            # Register rating, if not cold
            ratings_dict[user][item] = rating

        return dict(ratings_dict)

    def _build_mapped_dict(self, public_dict):
        private_dict = {}
        if public_dict is None:
            return None

        for user, items in public_dict.items():
            mapped_user = self._u_map.get(user)

            new_items = {}
            for i, v in items.items():
                mapped_item = self._i_map.get(i)
                new_items[mapped_item] = v

            private_dict[mapped_user] = new_items

        return private_dict

    def _get_users_and_items(self, private=False):
        ratings_dict = self._train_dict if not private else self._i_train_dict

        users = list(ratings_dict.keys())
        item_set = set()

        for user_ratings in ratings_dict.values():
            item_set.update(user_ratings.keys())

        items = list(item_set)

        return users, items

    def _get_triples(self):
        users, items, ratings = [], [], []
        for u, item_list in self._i_train_dict.items():
            for i, r in item_list.items():
                users.append(u)
                items.append(i)
                ratings.append(r)
        return users, items, ratings

    def training_dataloader(self, sampler_cls, seed=42, **kwargs):
        cache_key = sampler_cls.__name__
        if cache_key in self._cached_samplers:
            return self._cached_samplers[cache_key]

        if kwargs.get('transactions') is not None:
            transactions = kwargs.pop('transactions')
        else:
            transactions = self.transactions

        sampler = sampler_cls(
            users=self.users,
            items=self.items,
            train_dict=self.get_train_dict(private=True),
            transactions=transactions,
            seed=seed,
            **kwargs
        )
        samples = sampler.initialize()
        tensors = tuple(torch.tensor(x, dtype=torch.long) for x in samples)

        dataset = TensorDataset(*tensors)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._cached_samplers[cache_key] = dataloader

        return dataloader

    def neg_eval_dataloader(self, batch_size):
        dataloader = NegEvalDataLoader(self.neg_eval_dataset, batch_size)
        return dataloader

    def full_eval_dataloader(self, batch_size):
        dataloader = FullEvalDataloader(self.full_eval_dataset, batch_size)
        return dataloader

    @cached_property
    def sp_i_train(self):
        return self.sp_i_train_ratings.astype(bool).astype('float32')

    @cached_property
    def sp_i_train_tensor(self):
        coo = self.sp_i_train.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        return SparseTensor(row=row, col=col, sparse_sizes=coo.shape)

    def get_mappings(self):
        return self._u_map, self._i_map

    def get_inverse_mappings(self):
        return self.users, self.items

    def get_train_dict(self, private=False):
        return self._train_dict if not private else self._i_train_dict

    def get_test_dict(self, private=False):
        return self._test_dict if not private else self._i_test_dict

    def get_val_dict(self, private=False):
        return self._val_dict if not private else self._i_val_dict

    def get_positive_items(self):
        users = sorted(list(self._i_train_dict.keys()))

        pos, val, test = [], [], []

        # Local cache to speed up computation
        train = self._i_train_dict
        test_dict = self._i_test_dict
        val_dict = self._i_val_dict

        has_val = val_dict is not None

        for u in users:
            items_train = train.get(u, ())
            items_test = test_dict.get(u, ())

            # Convert to set
            train_set = set(items_train)
            test_set = set(items_test)

            # Add test set
            test.append(list(test_set))

            # Positives = train ∪ test (U val if present)
            if has_val:
                items_val = val_dict.get(u, ())
                val_set = set(items_val)
                val.append(list(val_set))

                all_items = train_set | test_set | val_set
            else:
                all_items = train_set | test_set

            pos.append(list(all_items))

        return pos, val, test

    def build_items_neighbour(self):
        row, col = self.sp_i_train.nonzero()
        edge_index = np.array([row, col])
        iu_dict = {i: edge_index[0, iu].tolist() for i, iu in
                   enumerate(list((edge_index[1] == i).nonzero()[0] for i in self.items))}
        return iu_dict

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

    def _align_with_training(self, side_information_data):
        """Alignment with training"""

        def equal(a, b, c):
            return len(a) == len(b) == len(c)

        users = set(self._train_dict.keys())
        items = set({k for a in self._train_dict.values() for k in a.keys()})
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

    def _annotate_side_information(self):
        """
        Attach useful mappings to side-information namespaces so CB/Hybrid models
        can consume them without re-building public/private mappings.
        """
        for _, side_ns in self.side_information.__dict__.items():
            mapped_users, mapped_items = side_ns.object.get_mapped()
            setattr(side_ns, "user_mapping", self._u_map)
            setattr(side_ns, "item_mapping", self._i_map)
            setattr(side_ns, "mapped_users", {u: self._u_map[u] for u in mapped_users if u in self._u_map})
            setattr(side_ns, "mapped_items", {i: self._i_map[i] for i in mapped_items if i in self._i_map})
            setattr(side_ns, "num_users", self.num_users)
            setattr(side_ns, "num_items", self.num_items)
            # Alignment strategy and materialization hints from loader
            setattr(side_ns, "alignment_mode", getattr(side_ns.object, "_alignment_mode", None))
            setattr(side_ns, "materialization", getattr(side_ns.object, "_materialization", None))

    def _log_side_information(self):
        for name, side_ns in self.side_information.__dict__.items():
            mapped_users, mapped_items = side_ns.object.get_mapped()
            missing_users = len(set(self._train_dict.keys()) - set(mapped_users))
            missing_items = len(set(self._i_map.keys()) - set(mapped_items))
            self.logger.info(
                "Side information aligned",
                extra={
                    "context": {
                        "source": name,
                        "users_in_side": len(mapped_users),
                        "items_in_side": len(mapped_items),
                        "missing_users_vs_train": missing_users,
                        "missing_items_vs_train": missing_items,
                        "alignment_mode": getattr(side_ns, "alignment_mode", None),
                        "materialization": getattr(side_ns, "materialization", None),
                    }
                },
            )
