"""
Module description:

"""


from typing import Tuple, Dict, Optional
from types import SimpleNamespace

import pandas as pd
import copy
import logging as pylog
from torch.utils.data import Dataset, DataLoader

from elliot.dataset.interactions import Interactions
from elliot.dataset.sessions import Sessions, EvalSessions
from elliot.dataset.fusion.fuser import FeatureFuser
from elliot.dataset.samplers_eval import NegativeSampler, NegEvalDataset, FullEvalDataset
from elliot.utils import logging
from elliot.utils.enums import SessionStrategy


class DataSet:
    """
    Load train and test dataset
    """
    train_set: Interactions
    eval_set: Interactions
    train_sessions: Sessions
    eval_sessions: Optional[EvalSessions]
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

        self._cached_datasets = {}
        self._eval_cache_key = None
        self._session_eval_cache_key = None

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

        self.eval_set = Interactions(
            dataframe=eval_data,
            name=self._evaluation_set,
            u_map=self._u_map,
            i_map=self._i_map,
            side_info_ns=self.side_information
        )

        self.train_sessions = Sessions(
            dataframe=self.train_set.dataframe,
            name="train",
            u_map=self._u_map,
            i_map=self._i_map,
            side_info_ns=self.side_information,
            sparse=self.train_set.sparse_ratings
        )

        self.eval_sessions = None
        self.session_only_evaluation = False

    def _build_eval_sessions(self):
        self.eval_sessions = EvalSessions(
            dataframe=self.eval_set.dataframe,
            u_map=self._u_map,
            i_map=self._i_map,
            inv_mappings=self.get_inverse_mappings(),
            n_items=self.train_set.dims[1]
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

    def get_eval_dataloader(
        self,
        batch_size: int = 1024,
        session_strategy: Optional[SessionStrategy] = None
    ) -> DataLoader:
        requested_session = session_strategy == SessionStrategy.SESSION_ONLY
        dataset_has_sessions = self.config.data_config.session_strategy == SessionStrategy.SESSION_ONLY

        if requested_session and not dataset_has_sessions:
            self.logger.warning(
                "Model requested SESSION_ONLY evaluation, but this dataset was loaded with the "
                "FLAT session strategy (no session segmentation was performed); "
                "falling back to FLAT evaluation."
            )
            requested_session = False

        is_session = requested_session
        self.session_only_evaluation = is_session

        cache_key = (
            "session_neg" if is_session and self.config.negative_sampling is not None else
            "session_full" if is_session else
            "neg" if self.config.negative_sampling is not None else
            "full"
        )

        if cache_key not in self._cached_datasets:
            dataset = self._build_eval_dataset()
            self._cached_datasets[cache_key] = dataset

        dataset = self._cached_datasets[cache_key]

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )

        return dataloader

    def _build_eval_dataset(self) -> Dataset:
        if self.session_only_evaluation:
            if self.eval_sessions is None:
                self._build_eval_sessions()

            sessions = self.eval_sessions
            owner = sessions.owner_users

            train_pos_per_user = self.train_set.get_positive_items()
            eval_pos_per_user = self.eval_set.get_positive_items()

            train_pos = [train_pos_per_user[u] for u in owner]
            # Negatives must never be an item the user has interacted with in
            # *any* of their eval sessions, not just this row's own session,
            # so exclusion uses the owning user's full eval-set aggregate.
            neg_exclusion_pos = [eval_pos_per_user[u] for u in owner]
            # The ground-truth positive for a session-only row is only that
            # session's own masked (last) item, never another session of the
            # same user.
            eval_pos = [[int(t)] for t in sessions.target_items]

            _, public_items = self.get_inverse_mappings()
            row_public_ids = sessions.row_public_ids

            num_users = sessions.n_sessions

            mappings = (
                {row_id: idx for idx, row_id in enumerate(row_public_ids)},
                self._i_map,
            )
            inv_mappings = (row_public_ids, public_items)

        else:
            num_users, num_items = self.train_set.dims
            train_pos = self.train_set.get_positive_items()
            eval_pos = self.eval_set.get_positive_items()
            neg_exclusion_pos = eval_pos

            mappings = self.get_mappings()
            inv_mappings = self.get_inverse_mappings()

        num_items = self.train_set.dims[1]

        if self.config.negative_sampling is not None:
            sampler = NegativeSampler(
                neg_sampling_config=self.config.negative_sampling,
                mappings=mappings,
                inv_mappings=inv_mappings,
                num_users=num_users,
                num_items=num_items,
                train_pos_items=train_pos,
                eval_pos_items=neg_exclusion_pos,
                evaluation_set=self._evaluation_set,
                fold_index=self.fold_index,
            )

            eval_dataset = NegEvalDataset(
                num_users=num_users,
                sampler=sampler,
                eval_pos_items=eval_pos,
                evaluation_set=self._evaluation_set,
                leave_one_out=self.config.negative_sampling.leave_one_out,
            )

        else:
            eval_dataset = FullEvalDataset(num_users=num_users)

        return eval_dataset

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
