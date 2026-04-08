"""
Module description:

"""


from typing import Tuple, Dict, Union, Optional
from types import SimpleNamespace

import pandas as pd
import copy
import logging as pylog
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from elliot.dataset.interactions import Interactions
from elliot.dataset.fusion.fuser import FeatureFuser
from elliot.negative_sampling import NegativeSampler
from elliot.utils import logging


class NegEvalDataset(Dataset):
    def __init__(self, num_users, sampler, eval_pos_items, evaluation_set="test", leave_one_out=False):
        self.num_users = num_users
        self.leave_one_out = leave_one_out
        self._evaluation_set = evaluation_set

        eval_neg_items = sampler.sample()

        self.eval_items = self._add_indices(eval_neg_items, eval_pos_items)

    def _add_indices(self, neg, pos):
        """Add test or validation samples to the sampled negatives."""
        if not neg:
            return None

        final_items = []
        iter_data = tqdm(
            total=len(neg),
            desc=f"Adding {self._evaluation_set} items to sampled negatives",
            leave=False,
        )

        with iter_data as t:
            for neg_u, pos_u in zip(neg, pos):
                if not neg_u:
                    pos_u = []
                elif self.leave_one_out:
                    pos_u = [pos_u[-1]] if pos_u else []

                final_items.append(torch.tensor(neg_u + pos_u))
                t.update(1)

        return final_items

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return idx, self.eval_items[idx]

    @staticmethod
    def collate_fn(batch):
        user_indices, item_indices = zip(*batch)

        # User indices will be a list of ints, so we convert it
        user_indices = torch.tensor(list(user_indices))

        # We use the pad_sequence utility to pad item indices
        # in order to have all tensors of the same size
        item_indices = pad_sequence(
            item_indices,
            batch_first=True,
            padding_value=-1,
        )

        return user_indices, item_indices


class FullEvalDataset(Dataset):
    def __init__(self, num_users):
        self.num_users = num_users

    def __len__(self):
        return self.num_users

    def __getitem__(self, idx):
        return idx

    @staticmethod
    def collate_fn(batch):
        return torch.tensor(batch), None


class DataSet:
    """
    Load train and test dataset
    """
    train_set: Interactions
    eval_set: Interactions
    side_information: Dict[str, SimpleNamespace]

    def __init__(
        self,
        config,
        train_data,
        eval_data,
        side_info_data,
        evaluation_set: str = "test",
        fold_index: Tuple[int, Optional[int]] = (0, None),
        *args,
        **kwargs
    ):
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

        self.fold_index = fold_index

        self._evaluation_set = evaluation_set

        self._handle_train_set(train_data, side_info_data)
        self._handle_eval_set(eval_data)

        self._cached_datasets = {}
        self._eval_cache_key = None

    def _handle_train_set(
        self,
        train_data: Union[pd.DataFrame, Interactions],
        side_info_data
    ):
        if isinstance(train_data, pd.DataFrame):
            self._users = (
                train_data["userId"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
            self._items = (
                train_data["itemId"]
                .drop_duplicates()
                .sort_values()
                .tolist()
            )
        else:
            self._users, self._items = train_data.get_users_items()

        self._p_users = list(range(len(self._users)))
        self._p_items = list(range(len(self._items)))

        self._u_map = {user: k for k, user in zip(self._p_users, self._users)}
        self._i_map = {item: k for k, item in zip(self._p_items, self._items)}

        if self.config.align_side_with_train:
            side_objs = self._align_with_training(side_info_data)
        else:
            side_objs = side_info_data

        self.side_information = {k: v.create_namespace() for k, v in side_objs.items()}

        if self.side_information:
            self._annotate_side_information()
            self._log_side_information()
            self.fuser = FeatureFuser(self.side_information)
        else:
            # self.side_information = None
            self.fuser = None

        if isinstance(train_data, pd.DataFrame):
            self.train_set = Interactions(
                dataframe=train_data,
                name="train",
                u_map=self._u_map,
                i_map=self._i_map,
                side_info_ns=self.side_information
            )

            num_users, num_items = self.train_set.dims
            transactions = self.train_set.transactions
            sparsity = 1 - (transactions / (num_users * num_items))

            self.logger.info(
                f"Statistics\t"
                f"Users:\t{num_users}\t"
                f"Items:\t{num_items}\t"
                f"Transactions:\t{transactions}\t"
                f"Sparsity:\t{sparsity}"
            )
        else:
            self.train_set = train_data

    def _handle_eval_set(self, eval_data):
        self.eval_set = Interactions(
            dataframe=eval_data,
            name=self._evaluation_set,
            u_map=self._u_map,
            i_map=self._i_map,
            side_info_ns=self.side_information
        )

    def _align_with_training(self, side_information_data):
        """Alignment with training"""

        def equal(a, b, c):
            return len(a) == len(b) == len(c)

        side_objs = copy.deepcopy(side_information_data)

        users, items = self._users.copy(), self._items.copy()
        users_items = []

        for v in side_objs.values():
            users_items.append(v.get_mapped())

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
                for v in side_objs.values():
                    v.filter(users, items)
                    new_users_items.append(v.get_mapped())
                users_items = new_users_items

        return side_objs

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
            # setattr(side_ns, "num_users", self.num_users)
            # setattr(side_ns, "num_items", self.num_items)
            # Alignment strategy and materialization hints from loader
            setattr(side_ns, "alignment_mode", getattr(side_ns.object, "_alignment_mode", None))
            setattr(side_ns, "materialization", getattr(side_ns.object, "_materialization", None))

    def _log_side_information(self):
        for name, side_ns in self.side_information.__dict__.items():
            mapped_users, mapped_items = side_ns.object.get_mapped()
            missing_users = len(set(self.train_set.get_dict().keys()) - set(mapped_users))
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

    def get_eval_dataloader(self, batch_size=1024):
        cache_key = self._eval_cache_key

        if cache_key is None:
            num_users, num_items = self.train_set.dims

            if self.config.negative_sampling is not None:
                train_pos = self.train_set.get_positive_items()
                eval_pos = self.eval_set.get_positive_items()

                sampler = NegativeSampler(
                    neg_sampling_config=self.config.negative_sampling,
                    mappings=self.get_mappings(),
                    inv_mappings=self.get_inverse_mappings(),
                    num_users=num_users,
                    num_items=num_items,
                    train_pos_items=train_pos,
                    eval_pos_items=eval_pos,
                    evaluation_set=self._evaluation_set,
                    fold_index=self.fold_index
                )

                eval_dataset = NegEvalDataset(
                    num_users=num_users,
                    sampler=sampler,
                    eval_pos_items=eval_pos,
                    evaluation_set=self._evaluation_set,
                    leave_one_out=self.config.negative_sampling.leave_one_out
                )
                cache_key = "neg"
            else:
                eval_dataset = FullEvalDataset(num_users=num_users)
                cache_key = "full"

            self._cached_datasets[cache_key] = eval_dataset
            self._eval_cache_key = cache_key

        dataset = self._cached_datasets[cache_key]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False
        )

        return dataloader

    def get_mappings(self):
        return self._u_map, self._i_map

    def get_inverse_mappings(self):
        return self._users, self._items

    def get_users_items(self, private=False):
        return (self._users, self._items) if not private else (self._p_users, self._p_items)

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
