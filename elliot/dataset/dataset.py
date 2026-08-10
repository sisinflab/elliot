from typing import List, Tuple, Dict, Any, Optional
import pandas as pd
import logging as pylog
from torch.utils.data import Dataset, DataLoader

from elliot.dataset.interactions import Interactions
from elliot.dataset.sessions import Sessions, EvalSessions
from elliot.dataset.samplers_eval import NegativeSampler, NegEvalDataset, FullEvalDataset
from elliot.utils import logging
from elliot.utils.enums import SessionStrategy


class DataSet:
    """Train/eval pair for one fold. Wraps the fold's `Interactions` (train and eval)
    and `Sessions` (train) views, and lazily builds the evaluation dataloader --
    full-ranking or negative-sampled via `NegativeSampler`, optionally scoped to
    `EvalSessions` for SESSION_ONLY evaluation.
    """

    train_set: Interactions
    eval_set: Interactions
    train_sessions: Sessions
    eval_sessions: Optional[EvalSessions] = None

    def __init__(
        self,
        config,
        train_data,
        eval_data,
        side_info,
        evaluation_set: str = "test",
        fold_index: Tuple[int, Optional[int]] = (0, None),
        *args,
        **kwargs
    ):
        self.logger = logging.get_logger(
            self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG
        )

        # Initializing variables
        self.config = config
        self.args = args
        self.kwargs = kwargs

        self._users: List[Any] = []
        self._items: List[Any] = []
        self._p_users: List[Any] = []
        self._p_items: List[Any]  = []
        self._u_map: Dict[Any, int] = {}
        self._i_map: Dict[Any, int] = {}
        self._evaluation_set = evaluation_set
        self._cached_datasets = {}

        self.fold_index = fold_index
        self.session_only_evaluation = False

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

        if isinstance(train_data, pd.DataFrame):
            self.train_set = Interactions(
                dataframe=train_data,
                name="train",
                u_map=self._u_map,
                i_map=self._i_map,
                inv_mappings=(self._users, self._items),
                side_info=side_info
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

        if side_info is not None:
            for name, loader in side_info.items():
                mapped_users, mapped_items = loader.get_mapped()
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
                            "alignment_mode": getattr(loader, "alignment", None),
                            "materialization": getattr(loader, "materialization", None),
                        }
                    },
                )

        self.eval_set = Interactions(
            dataframe=eval_data,
            name=self._evaluation_set,
            u_map=self._u_map,
            i_map=self._i_map,
            inv_mappings=(self._users, self._items),
            side_info=side_info
        )

        self.train_sessions = Sessions(
            dataframe=self.train_set.dataframe,
            name="train",
            u_map=self._u_map,
            i_map=self._i_map,
            sparse=self.train_set.sparse_ratings
        )

    def _build_eval_sessions(self):
        self.eval_sessions = EvalSessions(
            dataframe=self.eval_set.dataframe,
            u_map=self._u_map,
            i_map=self._i_map,
            inv_mappings=self.get_inverse_mappings(),
            n_items=self.train_set.dims[1]
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
