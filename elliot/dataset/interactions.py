from typing import Tuple, Dict
from types import SimpleNamespace
import copy
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from collections import defaultdict
from functools import cached_property
from tqdm import tqdm

from elliot.dataset.samplers.base_sampler import build_dataset
from elliot.utils.registry import sampler_registry


class Interactions:
    dataframe: pd.DataFrame
    dims: Tuple[int, int]
    transactions: int
    sparse_ratings: csr_matrix
    sparse: csr_matrix
    sparse_tensor: SparseTensor
    side_information: Dict[str, SimpleNamespace]

    def __init__(
        self,
        dataframe,
        name,
        u_map,
        i_map,
        side_info_ns
    ):
        self.name = name

        self._dataframe = dataframe
        self._u_map = u_map
        self._i_map = i_map

        self._cold_items = set()
        self._cold_users = set()

        self._dict = self._build_dict()
        self._p_dict = self._build_mapped_dict()

        self._users, self._items = self._get_users_and_items()
        self._n_users = len(self._users)
        self._n_items = len(self._items)

        self._transactions = sum(len(v) for v in self._dict.values())

        if name == "train":
            _ = self.sparse_ratings

        self.side_information = self._align_side_info(side_info_ns)

        self._cached_datasets = {}

    def _build_dict(self, skip_cold_users_items=True):
        """Conversion to Dictionary"""
        ratings_dict = defaultdict(dict)

        data = self._dataframe
        users, items, ratings = data["userId"], data["itemId"], data["rating"]

        iter_df = tqdm(
            zip(users, items, ratings),
            total=len(users),
            desc=f"Building ratings dict for {self.name}",
            leave=False
        )

        for user, item, rating in iter_df:
            if skip_cold_users_items:
                # Cold user?
                if user not in self._u_map:
                    self._cold_users.add(user)
                    # And cold item?
                    if item not in self._i_map:
                        self._cold_items.add(item)
                    continue

                ratings_dict.setdefault(user, {})

                # Cold item?
                if item not in self._i_map:
                    self._cold_items.add(item)
                    continue

            # Register rating, if not cold
            ratings_dict[user][item] = rating

        return dict(ratings_dict)

    def _build_mapped_dict(self):
        private_dict = {}

        if self._u_map is None and self._i_map is None:
            return self._dict

        for user, items in self._dict.items():
            mapped_user = self._u_map.get(user)

            new_items = {}
            for i, v in items.items():
                mapped_item = self._i_map.get(i)
                new_items[mapped_item] = v

            private_dict[mapped_user] = new_items

        return private_dict

    def _get_users_and_items(self, private=False):
        ratings_dict = self._dict if not private else self._p_dict

        users = list(ratings_dict.keys())
        item_set = set()

        for user_ratings in ratings_dict.values():
            item_set.update(user_ratings.keys())

        items = sorted(list(item_set))

        return users, items

    def _align_side_info(self, side_info_ns):
        new_side_info_ns = {}

        for k, v in side_info_ns.items():
            side_obj = copy.deepcopy(v.object)
            side_obj.filter(self._users, self._items)
            new_side_info_ns[k] = side_obj.create_namespace()

        return new_side_info_ns

    def get_dataloader(self, sampler_name, batch_size=1024, seed=42, **kwargs):
        if kwargs.get('transactions') is not None:
            transactions = kwargs.pop('transactions')
        else:
            transactions = self._transactions

        if (sampler_name not in self._cached_datasets or
            len(self._cached_datasets[sampler_name]) != transactions):

            users, items = self._get_users_and_items(private=True)

            sampler = sampler_registry.get(
                name=sampler_name,
                train_dict=self.get_dict(private=True),
                transactions=transactions,
                users=users,
                items=items,
                n_users=self._n_users,
                n_items=self._n_items,
                seed=seed,
                **kwargs
            )

            dataset = build_dataset(sampler)

            self._cached_datasets[sampler_name] = dataset

        dataset = self._cached_datasets[sampler_name]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=getattr(dataset, 'collate_fn', None),
            shuffle=True
        )

        return dataloader

    def build_items_neighbour(self):
        row, col = self.sparse.nonzero()
        edge_index = np.array([row, col])
        iu_dict = {i: edge_index[0, iu].tolist() for i, iu in
                   enumerate(list((edge_index[1] == i).nonzero()[0] for i in self._items))}
        return iu_dict

    @cached_property
    def sparse_ratings(self):
        return self._to_sparse()

    @cached_property
    def sparse(self):
        return self.sparse_ratings.astype(bool).astype('float32')

    @cached_property
    def sparse_tensor(self):
        coo = self.sparse.tocoo()
        row = torch.tensor(coo.row, dtype=torch.long)
        col = torch.tensor(coo.col, dtype=torch.long)
        return SparseTensor(row=row, col=col, sparse_sizes=coo.shape)

    @property
    def dataframe(self):
        return self._dataframe

    @property
    def dims(self):
        return self._n_users, self._n_items

    @property
    def transactions(self):
        return self._transactions

    def get_users_items(self):
        return self._users, self._items

    def get_dict(self, private=False):
        return self._dict if not private else self._p_dict

    def get_positive_items(self):
        users = sorted(list(self._p_dict.keys()))

        pos = []
        for u in users:
            items_train = self._p_dict.get(u, ())
            train_set = list(set(items_train))
            pos.append(list(train_set))

        return pos

    def _get_triples(self):
        users, items, ratings = [], [], []
        for u, item_list in self._p_dict.items():
            for i, r in item_list.items():
                users.append(u)
                items.append(i)
                ratings.append(r)
        return users, items, ratings

    def _to_sparse(self):
        rows, cols, data = self._get_triples()
        return csr_matrix(
            (data, (rows, cols)), dtype=float, shape=(len(self._u_map), len(self._i_map))
        )
