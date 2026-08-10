from typing import Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from scipy.sparse import csr_matrix
from collections import defaultdict
from functools import cached_property
from tqdm import tqdm

from elliot.dataset.modular_loaders.remap import (
    remap_embedding_payload,
    remap_pair_payload,
    remap_text_payload,
)
from elliot.dataset.modular_loaders.formats import EmbeddingPayload, TextPayload
from elliot.dataset.samplers.base_sampler import build_dataset
from elliot.utils.enums import EntityAxis
from elliot.utils.registry import sampler_registry


class Interactions:
    """Unordered user-item view of one split (train or eval), parallel to `Sessions`
    (which is ordered). Builds the ratings dict/sparse matrix once and exposes
    per-fold sampler dataloaders plus this split's own private view of any side
    information.
    """

    dataframe: pd.DataFrame
    dims: Tuple[int, int]
    transactions: int
    sparse_ratings: csr_matrix
    sparse: csr_matrix
    sparse_tensor: SparseTensor

    def __init__(
        self,
        dataframe,
        name,
        u_map,
        i_map,
        inv_mappings,
        side_info
    ):
        # Initializing variables
        self.name = name

        self._dataframe = dataframe
        self._side_info = side_info

        # Set mappings
        self._u_map = u_map
        self._i_map = i_map
        self._u_map_inv, self._i_map_inv = inv_mappings

        self._cold_items = set()
        self._cold_users = set()
        self._cached_datasets = {}

        self._dict = self._build_dict()
        self._p_dict = self._build_mapped_dict()

        self._users, self._items = self._get_users_and_items()
        self._n_users = len(self._users)
        self._n_items = len(self._items)

        self._transactions = sum(len(v) for v in self._dict.values())

        if name == "train":
            _ = self.sparse_ratings

        # Shared by reference -- never copied/mutated. The materialized payload itself
        # lives on `self._side_info` (cached once for the whole experiment); this
        # instance's own `_side_info_cache` just holds a reference to it.
        self._side_info_cache = (
            {loader_name: None for loader_name in self._side_info} if self._side_info else {}
        )

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

    @property
    def side_information(self):
        return self._side_info_cache

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

    def get_loader(self, name):
        """Return the raw `AbstractLoader` instance registered under `name` on the
        shared `SideInformation` -- e.g. to inspect `get_mapped()`/`alignment` directly,
        as opposed to `get_side_info()`, which returns its *materialized, per-fold-remapped*
        payload.
        """
        return self._side_info[name]

    def get_side_info(self, name):
        """Return this loader's payload, remapped into *this fold's* private-id view
        (see `_to_private_view()`/`elliot.dataset.modular_loaders.adapters.
        remap_embedding_payload`) and cached here so a second request skips even that.

        The shared, public-id-keyed payload itself is still materialized/cached only
        once for the whole experiment, on `SideInformation` (`get_payload()`); this
        just builds -- once per fold -- the cheap-to-lazy-but-sometimes-real-copy view
        on top of it. This instance registers itself (weakly) with `SideInformation`
        so its own cached view is dropped, via `forget_side_info()`, once nothing in
        the experiment still needs this loader (`SideInformation.marked_as_done()`).
        """
        if self._side_info_cache.get(name) is None:
            payloads = self._side_info.get_payload(name)
            self._side_info_cache[name] = {
                key: self._to_private_view(name, key, payload) for key, payload in payloads.items()
            }
            self._side_info.register_private_view(name, self)
        return self._side_info_cache[name]

    def forget_side_info(self, name: str) -> None:
        """Drop this instance's own cached private view for `name`. Called either by
        the shared `SideInformation` once every model declaring the loader is done
        with every fold (see `SideInformation.marked_as_done()`), so a per-fold copy
        doesn't linger for the rest of the run just because this fold's `Interactions`
        stays alive, or directly by this instance itself to drop its own reference
        early -- the shared, centrally-cached payload itself is untouched either way,
        so a later `get_side_info()` call just rebuilds this instance's view from it.
        """
        self._side_info_cache.pop(name, None)

    def _to_private_view(self, name, key, payload):
        axis = self._side_info[name].entity_axis.get(key, EntityAxis.NONE)
        if axis is EntityAxis.NONE:
            return payload
        if axis is EntityAxis.PAIR:
            return remap_pair_payload(payload, self._u_map, self._i_map)

        mapping = self._u_map_inv if axis is EntityAxis.USER else self._i_map_inv
        if isinstance(payload, EmbeddingPayload):
            return remap_embedding_payload(payload, mapping)
        if isinstance(payload, TextPayload):
            return remap_text_payload(payload, mapping)
        return payload

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
