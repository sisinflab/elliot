from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import logging as pylog
from torch.utils.data import Dataset, DataLoader

from elliot.dataset.modular_loaders import SideInformation
from elliot.dataset.interactions import Interactions
from elliot.dataset.sessions import Sessions, EvalSessions
from elliot.dataset.samplers_eval import NegativeSampler, NegEvalDataset, FullEvalDataset
from elliot.namespace import ExperimentConfig
from elliot.utils import logging
from elliot.utils.enums import SessionStrategy


class DataSet:
    """Train/eval pair for one fold. Wraps the fold's `Interactions` (train and eval)
    and `Sessions` (train) views, and lazily builds the evaluation dataloader,
    full-ranking or negative-sampled via `NegativeSampler`, optionally scoped to
    `EvalSessions` for SESSION_ONLY evaluation.

    Args:
        config (ExperimentConfig): Experiment configuration object.
        train_data (Union[pd.DataFrame, Interactions]): Train interactions, either as a raw
            DataFrame (built fresh) or an already-built `Interactions` (shared, e.g.
            across validation folds of the same test fold).
        eval_data (pd.DataFrame): Eval (test/validation) interactions.
        side_info (SideInformation, optional): Shared, centrally-cached side-information handle.
            Defaults to None.
        evaluation_set (str): Name of this fold's eval split ("test" or "validation").
            Defaults to "test".
        fold_index (Tuple[int, Optional[int]]): Tuple containing the complete fold index.
            Defaults to `(0, None)`.
    """

    train_set: Interactions
    eval_set: Interactions
    train_sessions: Sessions
    eval_sessions: Optional[EvalSessions] = None

    def __init__(
        self,
        config: ExperimentConfig,
        train_data: Union[pd.DataFrame, Interactions],
        eval_data: pd.DataFrame,
        side_info: Optional[SideInformation] = None,
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

        # Negatives sampled once per train user and cached, so that models evaluated
        # under different session strategies (FLAT and SESSION_ONLY) on this same
        # fold are always scored against the exact same negative items. Built lazily,
        # on the first call to `_get_user_negatives()`
        self._user_negatives: Optional[List[List[int]]] = None

        # Discover this fold's (user, item) domain from the raw train DataFrame, or
        # reuse an already-built `Interactions`' own domain when train is shared
        # (e.g. across validation folds of the same test fold)
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

        # Assign each public user/item id a private, 0-based index
        self._p_users = list(range(len(self._users)))
        self._p_items = list(range(len(self._items)))

        self._u_map = {user: k for k, user in zip(self._p_users, self._users)}
        self._i_map = {item: k for k, item in zip(self._p_items, self._items)}

        # Build the train view from raw data, or reuse an already-built one
        if isinstance(train_data, pd.DataFrame):
            self.train_set = Interactions(
                dataframe=train_data,
                name="train",
                mappings=self.get_mappings(),
                inv_mappings=self.get_inverse_mappings(),
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

        # Log each loader's user/item coverage against this fold's train domain
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

        # Build the eval view, sharing this fold's (user, item) mappings with train
        self.eval_set = Interactions(
            dataframe=eval_data,
            name=self._evaluation_set,
            mappings=self.get_mappings(),
            inv_mappings=self.get_inverse_mappings(),
            side_info=side_info
        )

        # Build the ordered, per-user/session view of train for sequential models
        self.train_sessions = Sessions(
            dataframe=self.train_set.dataframe,
            name="train",
            mappings=self.get_mappings(),
            sparse=self.train_set.sparse_ratings
        )

    def _build_eval_sessions(self):
        """Lazily build `self.eval_sessions`, this fold's eval-side `EvalSessions`
        view, used for SESSION_ONLY evaluation. Called at most once per fold, on the
        first request that actually needs it (see `_build_eval_dataset()`).
        """
        self.eval_sessions = EvalSessions(
            dataframe=self.eval_set.dataframe,
            mappings=self.get_mappings(),
            inv_mappings=self.get_inverse_mappings(),
            n_items=self.train_set.dims[1]
        )

    def get_eval_dataloader(
        self,
        batch_size: int = 1024,
        session_strategy: Optional[SessionStrategy] = None
    ) -> DataLoader:
        """Build (or reuse a cached) evaluation dataloader for this fold, honoring the
        configured negative-sampling and session strategies.

        Args:
            batch_size (int): Batch size for the returned dataloader. Defaults to 1024.
            session_strategy (SessionStrategy, optional): SESSION_ONLY to request
                per-eval-session evaluation; falls back to FLAT (with a warning) if the
                dataset itself wasn't loaded with SESSION_ONLY segmentation. Defaults
                to None (FLAT).

        Returns:
            DataLoader: The (possibly cached) evaluation dataloader.
        """
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

        # One cached dataset per (session strategy, negative-sampling) combination,
        # since either one changes what `_build_eval_dataset()` produces.
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
        """Build the evaluation `Dataset` for the current `session_only_evaluation`
        flag and negative-sampling configuration: full-ranking or negative-sampled
        (drawn from `_get_user_negatives()`'s per-user cache), scoped either to the
        whole train/eval user domain or to `EvalSessions`' per-session rows.

        Returns:
            Dataset: The evaluation dataset, ready to be wrapped in a `DataLoader`.
        """
        # Session-scoped: one row per eval session, targeting its own masked item
        if self.session_only_evaluation:
            if self.eval_sessions is None:
                self._build_eval_sessions()

            sessions = self.eval_sessions
            owner = sessions.owner_users

            # The ground-truth positive for a session-only row is only that
            # session's own masked (last) item, never another session of the
            # same user.
            eval_pos = [[int(t)] for t in sessions.target_items]

            num_users = sessions.n_sessions

        # Flat: one row per train user, targeting its whole eval positive set
        else:
            num_users = self.train_set.dims[0]
            eval_pos = self.eval_set.get_positive_items()

        # Negative-sampled evaluation, drawing from the per-user cache
        if self.config.negative_sampling is not None:
            if self._user_negatives is None:
                self.sample_user_negatives()
            user_negatives = self._user_negatives

            # Session rows don't own a private negative sample of their own: every
            # row broadcasts its owning user's single, once-sampled negative list
            eval_neg_items = (
                [user_negatives[u] for u in owner] if self.session_only_evaluation else user_negatives
            )

            eval_dataset = NegEvalDataset(
                num_users=num_users,
                eval_neg_items=eval_neg_items,
                eval_pos_items=eval_pos,
                evaluation_set=self._evaluation_set,
                leave_one_out=self.config.negative_sampling.leave_one_out,
            )

        # Full-ranking evaluation: no negatives, every item is a candidate
        else:
            eval_dataset = FullEvalDataset(num_users=num_users)

        return eval_dataset

    def sample_user_negatives(self):
        """Lazily sample (and cache) this fold's negative items once, over the whole
        train-user domain, regardless of which session strategy is requested first.
        Models evaluated with FLAT and SESSION_ONLY on this same fold must be scored
        against identical negatives per user for their metrics to be comparable, so
        `_build_eval_dataset()` never re-samples: it always reads from (and broadcasts)
        this single cache.
        """
        if self.config.negative_sampling is not None:
            sampler = NegativeSampler(
                neg_sampling_config=self.config.negative_sampling,
                mappings=self.get_mappings(),
                inv_mappings=self.get_inverse_mappings(),
                num_users=self.train_set.dims[0],
                num_items=self.train_set.dims[1],
                train_pos_items=self.train_set.get_positive_items(),
                eval_pos_items=self.eval_set.get_positive_items(),
                evaluation_set=self._evaluation_set,
                fold_index=self.fold_index,
            )
            self._user_negatives = sampler.sample()

    def get_mappings(self) -> Tuple[Dict[Any, int], Dict[Any, int]]:
        """Return this fold's (user, item) public-to-private id mappings.

        Returns:
            Tuple[Dict[Any, int], Dict[Any, int]]: The (user, item) mappings.
        """
        return self._u_map, self._i_map

    def get_inverse_mappings(self) -> Tuple[List[Any], List[Any]]:
        """Return this fold's (user, item) private-to-public id mappings, as lists
        indexed by private id.

        Returns:
            Tuple[List[Any], List[Any]]: The (user, item) inverse mappings.
        """
        return self._users, self._items

    def get_users_items(self, private: bool = False) -> Tuple[List[Any], List[Any]]:
        """Return this fold's (user, item) domain.

        Args:
            private (bool): If True, return private (index-space) ids; otherwise
                public ids. Defaults to False.

        Returns:
            Tuple[List[Any], List[Any]]: The (users, items) in this fold's domain.
        """
        return (self._users, self._items) if not private else (self._p_users, self._p_items)
