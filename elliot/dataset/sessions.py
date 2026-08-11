from typing import Any, Dict, List, Tuple
import copy
import numpy as np
import pandas as pd
import torch
from functools import cached_property
from scipy.sparse import csr_matrix
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from elliot.dataset.samplers.base_sampler import build_dataset
from elliot.utils import logging
from elliot.utils.enums import SessionStrategy
from elliot.utils.registry import sampler_registry


class Sessions:
    """Ordered, per-user/per-session view of the train set, parallel to
    `Interactions` (which is unordered). Supports two strategies for turning
    a user's train history into sequences:

    - FLAT: one sequence per user, the whole train history.
    - SESSION_ONLY: one sequence per (user, session); if the dataframe carries
      no `sessionId` column, every user has exactly one implicit session, so
      this degenerates to FLAT.

    Args:
        dataframe (pd.DataFrame): Train interactions, with at least 'userId', 'itemId',
            and 'timestamp' columns (and, optionally, 'sessionId').
        name (str): Name of this split (always "train").
        mappings (Tuple[Dict[Any, int], Dict[Any, int]]): (user, item) mappings from
            public ids to private indices.
        sparse (csr_matrix): Train ratings matrix, shared with `Interactions` and
            deep-copied here so sorting its indices never mutates the original.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        name: str,
        mappings: Tuple[Dict[Any, int], Dict[Any, int]],
        sparse: csr_matrix
    ):
        self.logger = logging.get_logger(self.__class__.__name__)

        # Initializing variables
        self.name = name

        self._u_map, self._i_map = mappings
        self._n_users = len(self._u_map)
        self._n_items = len(self._i_map)
        self._sparse = copy.deepcopy(sparse)

        self._cached_datasets: Dict[Tuple[str, SessionStrategy], Dataset] = {}

        self._sparse.sort_indices()
        self._build_tape(dataframe)

    def _build_tape(self, dataframe: pd.DataFrame) -> None:
        """Flatten `dataframe` into a single item tape, globally sorted by (user,
        session, timestamp), plus the per-user and per-session boundary arrays used to
        slice it into sequences.

        Args:
            dataframe (pd.DataFrame): Train interactions, with at least 'userId',
                'itemId', and 'timestamp' columns (and, optionally, 'sessionId').
        """
        df = dataframe[["userId", "itemId", "timestamp"]].copy()
        df["__u"] = df["userId"].map(self._u_map).to_numpy()
        df["__i"] = df["itemId"].map(self._i_map).to_numpy()
        self._has_sessions = "sessionId" in dataframe.columns
        df["__s"] = dataframe["sessionId"].to_numpy() if self._has_sessions else 0

        df = df.sort_values(["__u", "__s", "timestamp"], kind="stable")

        self._flat_users = df["__u"].to_numpy()
        self._flat_items = df["__i"].to_numpy()
        session_col = df["__s"].to_numpy()

        n = len(self._flat_items)

        # Per-user boundary offsets into the flat tape (users are contiguous, since
        # the tape is sorted by user first)
        self._user_offsets = np.searchsorted(self._flat_users, np.arange(self._n_users + 1))

        if n:
            # A new session starts whenever the user or session column changes from
            # the previous tape position
            is_new_session = np.empty(n, dtype=bool)
            is_new_session[0] = True
            is_new_session[1:] = (
                (self._flat_users[1:] != self._flat_users[:-1]) |
                (session_col[1:] != session_col[:-1])
            )

            # Cumulative sum of session-start flags gives each tape position its
            # 0-based, globally unique session index
            self._flat_session = np.cumsum(is_new_session) - 1
            self._n_sessions = int(self._flat_session[-1]) + 1
        else:
            self._flat_session = np.array([], dtype=np.int64)
            self._n_sessions = 0

        # Per-session boundary offsets into the flat tape, plus each session's
        # owning user (the user at its first tape position)
        self._session_offsets = np.searchsorted(self._flat_session, np.arange(self._n_sessions + 1))
        self._session_owner = (
            self._flat_users[self._session_offsets[:-1]]
            if self._n_sessions else np.array([], dtype=np.int64)
        )

    def get_dataloader(
        self,
        sampler_name: str,
        strategy: SessionStrategy = SessionStrategy.FLAT,
        batch_size: int = 1024,
        seed: int = 42,
        **kwargs: Any
    ) -> DataLoader:
        """Build (or reuse a cached) dataloader over this split's sessions, for the
        given sampler and session strategy.

        Args:
            sampler_name (str): Name of the sampler registered in `sampler_registry`.
            strategy (SessionStrategy): FLAT or SESSION_ONLY. Defaults to FLAT.
            batch_size (int): Batch size for the returned dataloader. Defaults to 1024.
            seed (int): Random seed forwarded to the sampler. Defaults to 42.
            **kwargs (Any): Additional keyword arguments forwarded to the sampler.

        Returns:
            DataLoader: The (possibly cached) dataloader for this sampler/strategy pair.
        """
        requested_session = strategy == SessionStrategy.SESSION_ONLY
        strategy = strategy if isinstance(strategy, SessionStrategy) else SessionStrategy(strategy)

        # Warn if SESSION_ONLY was requested but this split has no session boundaries
        if requested_session and not self._has_sessions:
            self.logger.warning(
                "Sampler requested SESSION_ONLY strategy, but this dataset was loaded with the "
                "FLAT session strategy (no session segmentation was performed); "
                "each user's full history will be treated as a single session."
            )

        cache_key = (sampler_name, strategy)

        # Build (and cache) a fresh sampler/dataset for this (sampler, strategy) pair
        if cache_key not in self._cached_datasets:
            sampler = sampler_registry.get(
                name=sampler_name,
                flat_items=self._flat_items,
                flat_users=self._flat_users,
                flat_session=self._flat_session,
                user_offsets=self._user_offsets,
                session_offsets=self._session_offsets,
                sparse=self._sparse,
                users=list(range(self._n_users)),
                items=list(range(self._n_items)),
                n_users=self._n_users,
                n_items=self._n_items,
                strategy=strategy,
                seed=seed,
                **kwargs
            )

            dataset = build_dataset(sampler)
            self._cached_datasets[cache_key] = dataset

        dataset = self._cached_datasets[cache_key]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=getattr(dataset, 'collate_fn', None),
            shuffle=True
        )

        return dataloader

    def get_history(self, user_indices: Any, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padded whole-train-history sequences for a batch of (private) user indices,
        plus lengths.

        Args:
            user_indices (Any): Iterable of private user indices.
            max_seq_len (int): Maximum sequence length; longer histories are
                truncated to their most recent `max_seq_len` items.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded item sequences (shape
                `(len(user_indices), max_seq_len)`) and their true lengths.
        """
        seqs, lens = [], []

        # Slice each user's history off the flat tape, keeping only its most recent items
        for u in user_indices:
            u = int(u)
            start, end = int(self._user_offsets[u]), int(self._user_offsets[u + 1])
            hist = self._flat_items[start:end]
            recent = hist[-max_seq_len:] if len(hist) else hist
            seqs.append(torch.tensor(recent, dtype=torch.long))
            lens.append(len(recent))

        if not seqs:
            return torch.empty((0, max_seq_len), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        # Pad every sequence up to the batch's fixed max length
        padded = pad_sequence(seqs, batch_first=True, padding_value=self._n_items)
        return padded, torch.tensor(lens, dtype=torch.long)

    @property
    def owner_users(self) -> np.ndarray:
        """Private (train-space) owning-user index per session."""
        return self._session_owner

    @property
    def dims(self) -> Tuple[int, int]:
        """Number of (users, items) in this split."""
        return self._n_users, self._n_items

    @property
    def n_sessions(self) -> int:
        """Total number of sessions in this split."""
        return self._n_sessions


class EvalSessions:
    """Lightweight, eval-side-only view of the eval (test/validation) set's
    own sessions, used for SESSION_ONLY evaluation. Each row is one eval
    session; its context is that session's own item prefix (leave-last-item-out),
    never train data or other sessions.

    Args:
        dataframe (pd.DataFrame): Eval interactions, with at least 'userId', 'itemId',
            and 'timestamp' columns (and, optionally, 'sessionId').
        mappings (Tuple[Dict[Any, int], Dict[Any, int]]): (user, item) mappings from
            public ids to private indices.
        inv_mappings (Tuple[List[Any], List[Any]]): Inverse (user, item) mappings from
            private indices back to public ids.
        n_items (int): Total number of items, used as the padding value for item
            sequences.
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        mappings: Tuple[Dict[Any, int], Dict[Any, int]],
        inv_mappings: Tuple[List[Any], List[Any]],
        n_items: int
    ):
        # Initializing variables
        self._u_map, self._i_map = mappings
        self._inv_u_map, self._inv_i_map = inv_mappings
        self._n_items = n_items

        self._build_tape(dataframe)

    def _build_tape(self, dataframe: pd.DataFrame) -> None:
        """Flatten `dataframe` into a single item tape, globally sorted by (user,
        session, timestamp), plus the per-row boundary and bookkeeping arrays used to
        slice it into one row per eval session.

        Args:
            dataframe (pd.DataFrame): Eval interactions, with at least 'userId',
                'itemId', and 'timestamp' columns (and, optionally, 'sessionId').
        """
        df = dataframe[["userId", "itemId", "timestamp"]].copy()
        df["__u"] = df["userId"].map(self._u_map)
        df["__s"] = dataframe["sessionId"].to_numpy() if "sessionId" in dataframe.columns else 0
        df["__i"] = df["itemId"].map(self._i_map)

        # Drop cold users/items (unseen in train, hence absent from u_map/i_map)
        df = df.dropna(subset=["__u", "__i"])
        df["__u"] = df["__u"].astype(np.int64)
        df["__i"] = df["__i"].astype(np.int64)

        df = df.sort_values(["__u", "__s", "timestamp"], kind="stable")

        flat_users = df["__u"].to_numpy()
        flat_items = df["__i"].to_numpy()
        session_col = df["__s"].to_numpy()

        n = len(flat_items)
        if n:
            # A new row (eval session) starts whenever the user or session column
            # changes from the previous tape position; cumulative sum of these
            # flags gives each tape position its 0-based, globally unique row index
            is_new_row = np.empty(n, dtype=bool)
            is_new_row[0] = True
            is_new_row[1:] = (flat_users[1:] != flat_users[:-1]) | (session_col[1:] != session_col[:-1])
            row_id = np.cumsum(is_new_row) - 1
            n_rows = int(row_id[-1]) + 1
        else:
            row_id = np.array([], dtype=np.int64)
            n_rows = 0

        # Per-row boundary offsets into the flat tape, plus each row's owning user
        # (the user at its first tape position)
        self._flat_items = flat_items
        self._row_starts = np.searchsorted(row_id, np.arange(n_rows + 1))
        self._n_rows = n_rows
        self._owner_users = flat_users[self._row_starts[:-1]] if n_rows else np.array([], dtype=np.int64)

        if n_rows:
            # 0-based index of each row among the rows owned by the same user, used
            # by `row_public_ids` to tell apart that user's several eval sessions
            _, first_idx, counts = np.unique(self._owner_users, return_index=True, return_counts=True)
            self._local_idx = np.arange(n_rows) - np.repeat(first_idx, counts)
        else:
            self._local_idx = np.array([], dtype=np.int64)

    def get_eval_context(self, row_indices: Any, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padded (all-but-last-item) context per requested row, plus lengths.

        Args:
            row_indices (Any): Iterable of eval-row indices.
            max_seq_len (int): Maximum context length; longer contexts are
                truncated to their most recent `max_seq_len` items.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded context sequences (shape
                `(len(row_indices), max_seq_len)`) and their true lengths.
        """
        seqs, lens = [], []

        # Slice off each row's last (masked) item, keeping only the most recent context
        for r in row_indices:
            r = int(r)
            start, end = int(self._row_starts[r]), int(self._row_starts[r + 1])
            context = self._flat_items[start:max(start, end - 1)]
            recent = context[-max_seq_len:] if len(context) else context
            seqs.append(torch.tensor(recent, dtype=torch.long))
            lens.append(len(recent))

        if not seqs:
            return torch.empty((0, max_seq_len), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        # Pad every context up to the batch's fixed max length
        padded = pad_sequence(seqs, batch_first=True, padding_value=self._n_items)
        return padded, torch.tensor(lens, dtype=torch.long)

    @cached_property
    def target_items(self) -> np.ndarray:
        """Private item id of each row's own masked (last) item: the single
        leave-last-item-out prediction target for that session, never
        another session of the same user."""
        ends = self._row_starts[1:] - 1
        return self._flat_items[ends]

    @cached_property
    def target_public_ids(self) -> List[Any]:
        """Public item id of each row's own masked (last) target item."""
        return [self._inv_i_map[int(i)] for i in self.target_items]

    @cached_property
    def owner_public_ids(self) -> List[Any]:
        """Public user id owning each eval row."""
        return [self._inv_u_map[int(u)] for u in self._owner_users]

    @cached_property
    def row_public_ids(self) -> List[str]:
        """Virtual public id per eval row,
        so `collector.py` doesn't collapse multiple sessions of the same user
        into a single `preds_dict` entry."""
        return [
            f"{self._inv_u_map[int(u)]}::s{int(li)}"
            for u, li in zip(self._owner_users, self._local_idx)
        ]

    @cached_property
    def owner_map(self) -> Dict[str, Any]:
        """Virtual public row id -> real public owning-user id.
        Used by `Evaluator` to average metrics within a user's sessions
        before averaging across users."""
        return dict(zip(self.row_public_ids, self.owner_public_ids))

    @property
    def owner_users(self) -> np.ndarray:
        """Private (train-space) owning-user index per eval row.
        Used by `collector.py` to mask against the owning user's train history
        rather than the row index itself."""
        return self._owner_users

    @property
    def n_sessions(self) -> int:
        """Total number of eval sessions (rows)."""
        return self._n_rows
