from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from functools import cached_property
from scipy.sparse import csr_matrix
from torch.nn.utils.rnn import pad_sequence

from elliot.namespace import UserFilterConfig
from elliot.utils import logging
from elliot.utils.enums import EvalTarget, SessionStrategy
from elliot.utils.folder import path_absolute, check_path
from elliot.utils.read import Reader


class EvalTargets:
    """Eval-side view of the eval (test/validation) set as one row per prediction
    target, shared by every model's evaluation. Each row pairs its target item(s)
    with the context known right before them:

    - FLAT: one sequence per user, its known `history` followed by its eval items;
      the target is picked among the eval items, the context is everything before it.
    - SESSION_ONLY: one sequence per eval session; the target is picked among the
      session's items (never its first one), the context is the session prefix.

    Rows with an empty context are dropped, as well as the rows of the users left out
    by the user filter file, if any. Rows are keyed by their owning user's public id,
    unless a user can own several of them (SESSION_ONLY or SUCCESSIVE): then each row
    gets a virtual `"{user}::{k}"` id instead.

    In experiments mixing SESSION_ONLY and other models (target LAST over a test set
    holding each user's last session, see `ExperimentConfig`), FLAT rows are built
    with `align_with_sessions`, so both kinds of models are scored on the same rows.

    Args:
        eval_data (pd.DataFrame): Eval interactions, with at least 'userId', 'itemId',
            and 'timestamp' columns (and, optionally, 'sessionId').
        history (pd.DataFrame): Interactions known before the eval ones (train), used
            as FLAT context.
        seen (csr_matrix): (users x items) matrix of the items known before the eval
            ones, used by `get_seen_mask()`.
        mappings (Tuple[Dict[Any, int], Dict[Any, int]]): (user, item) mappings from
            public ids to private indices.
        inv_mappings (Tuple[List[Any], List[Any]]): Inverse (user, item) mappings from
            private indices back to public ids.
        strategy (SessionStrategy): FLAT or SESSION_ONLY.
        target (EvalTarget): Which item(s) of each eval sequence are predicted.
        seed (int): Random seed for the RANDOM target. Defaults to 42.
        align_with_sessions (bool): FLAT only: also drop the rows whose target opens an
            eval session, which SESSION_ONLY evaluation can never score (they have no
            session prefix). Defaults to False.
        user_filter (UserFilterConfig, optional): The file listing the only users whose
            rows are kept (one id per line, in its `id_space`). Defaults to None (keeping
            every user), as does a config with no `path`.
    """

    def __init__(
        self,
        eval_data: pd.DataFrame,
        history: pd.DataFrame,
        seen: csr_matrix,
        mappings: Tuple[Dict[Any, int], Dict[Any, int]],
        inv_mappings: Tuple[List[Any], List[Any]],
        strategy: SessionStrategy,
        target: EvalTarget,
        seed: int = 42,
        align_with_sessions: bool = False,
        user_filter: Optional[UserFilterConfig] = None
    ):
        if strategy == SessionStrategy.SESSION_ONLY and target in (EvalTarget.ALL, EvalTarget.FIRST):
            raise ValueError(
                f"Evaluation target '{target.value}' is not available for SESSION_ONLY evaluation."
            )

        self.logger = logging.get_logger(self.__class__.__name__)
        self.reader = Reader(self.logger)

        # Initializing variables
        self._u_map, self._i_map = mappings
        self._inv_u_map, self._inv_i_map = inv_mappings
        self._n_items = len(self._i_map)
        self._seen = seen

        self.strategy = strategy
        self.target = target
        self.align_with_sessions = align_with_sessions
        self.user_filter = self._load_user_filter(user_filter)
        self.multi_row = strategy == SessionStrategy.SESSION_ONLY or target == EvalTarget.SUCCESSIVE

        self._build_rows(eval_data, history, np.random.default_rng(seed))

    def _load_user_filter(self, config: Optional[UserFilterConfig]) -> Optional[np.ndarray]:
        """Read the user filter file into the private indices of the only users to
        evaluate. Ids unknown to this fold are dropped.

        Args:
            config (UserFilterConfig, optional): The user filter configuration.

        Returns:
            Optional[np.ndarray]: The sorted private user indices, or None to evaluate
                every user (no file configured, or no known user in it).

        Raises:
            FileNotFoundError: If the configured user filter file doesn't exist.
        """
        if config is None or not config.path:
            return None

        path = path_absolute(config.path)
        if not check_path(path):
            raise FileNotFoundError(f"User filter file not found: {path}")

        # Each non-empty, non-comment line starts with a user id
        sep = config.reader.sep
        tokens = set(self.reader.read_lines(
            path,
            encoding="utf-8",
            parse_fn=lambda line: (line.strip().split(sep) if sep else line.split())[0].strip(),
            skip_fn=lambda line: not line.strip() or line.lstrip().startswith("#")
        ))

        # Map the ids to private indices, whatever their public type
        if config.id_space == "private":
            to_private = {str(idx): idx for idx in self._u_map.values()}
        else:
            to_private = {str(user): idx for user, idx in self._u_map.items()}
        users = {to_private[t] for t in tokens if t in to_private}

        if len(users) < len(tokens):
            self.logger.warning(
                "Dropped unknown users from the evaluation user filter",
                extra={"context": {"count": len(tokens) - len(users)}}
            )
        if not users:
            self.logger.warning("No known users in the evaluation user filter; evaluating every user.")
            return None

        self.logger.info("Evaluation user filter applied", extra={"context": {"users": len(users)}})
        return np.array(sorted(users), dtype=np.int64)

    def _build_rows(self, eval_data: pd.DataFrame, history: pd.DataFrame, rng: np.random.Generator):
        """Flatten every eval sequence into a single item tape, then pick each row's
        context start, target position(s) and owning user on it.

        Args:
            eval_data (pd.DataFrame): Eval interactions.
            history (pd.DataFrame): Interactions known before the eval ones.
            rng (np.random.Generator): Random generator for the RANDOM target.
        """
        is_session = self.strategy == SessionStrategy.SESSION_ONLY

        # FLAT sequences prepend each user's known history to its eval items
        parts = [self._to_private(eval_data, True)]
        if not is_session:
            parts.insert(0, self._to_private(history, False))

        df = pd.concat(parts, ignore_index=True)
        df = df.sort_values(["__u", "__e", "__s", "timestamp"], kind="stable")

        users = df["__u"].to_numpy(dtype=np.int64)
        sessions = df["__s"].to_numpy()
        is_eval = df["__e"].to_numpy(dtype=bool)
        self._flat_items = df["__i"].to_numpy(dtype=np.int64)
        self._flat_ratings = df["__r"].to_numpy(dtype=np.float64)

        # A new sequence starts whenever the user (or, for SESSION_ONLY, the session) changes
        n = len(users)
        is_new_seq = np.ones(n, dtype=bool)
        is_new_seq[1:] = users[1:] != users[:-1]
        if is_session:
            is_new_seq[1:] |= sessions[1:] != sessions[:-1]

        seq_start = np.flatnonzero(is_new_seq)
        seq_end = np.append(seq_start[1:], n)
        owner = users[seq_start]

        # Candidate target positions span [low, seq_end): a FLAT sequence's eval items
        # (which follow its history), or all of a session's items but its first one
        if is_session:
            low = seq_start + 1
        elif n:
            low = seq_start + np.add.reduceat((~is_eval).astype(np.int64), seq_start)
        else:
            low = seq_start
        high = seq_end

        # Eval items in [prefix_start, target) are seen as well (FLAT history already is)
        prefix_start = seq_start if is_session else low

        has_target = high > low
        seq_start, prefix_start = seq_start[has_target], prefix_start[has_target]
        low, high, owner = low[has_target], high[has_target], owner[has_target]

        seq_idx = np.arange(len(low))
        match self.target:
            case EvalTarget.SUCCESSIVE:
                seq_idx, cut = self._expand_ranges(low, high)
            case EvalTarget.ALL | EvalTarget.FIRST:
                cut = low
            case EvalTarget.LAST:
                cut = high - 1
            case EvalTarget.RANDOM:
                cut = low + (rng.random(len(low)) * (high - low)).astype(np.int64)

        # Each row's targets span [cut, target_end), its context [context_start, cut)
        target_end = high[seq_idx] if self.target == EvalTarget.ALL else cut + 1
        context_start = seq_start[seq_idx]

        keep = cut > context_start

        # Rows whose target opens an eval session have no session prefix, hence no
        # SESSION_ONLY counterpart: drop them to score FLAT rows on the same targets
        if self.align_with_sessions and not is_session:
            opens_session = is_eval.copy()
            opens_session[1:] &= is_new_seq[1:] | ~is_eval[:-1] | (sessions[1:] != sessions[:-1])
            keep &= ~opens_session[cut]

        # Only the filtered users are evaluated
        if self.user_filter is not None:
            keep &= np.isin(owner[seq_idx], self.user_filter)

        self._cut = cut[keep]
        self._target_end = target_end[keep]
        self._context_start = context_start[keep]
        self._prefix_start = prefix_start[seq_idx][keep]
        self._owner_users = owner[seq_idx][keep]
        self._n_rows = len(self._cut)

        # 0-based index of each row among the rows owned by the same user (rows are
        # sorted by owner), used by `row_public_ids` to tell them apart
        _, first_idx, counts = np.unique(self._owner_users, return_index=True, return_counts=True)
        self._local_idx = np.arange(self._n_rows) - np.repeat(first_idx, counts)

    def _to_private(self, dataframe: pd.DataFrame, is_eval: bool) -> pd.DataFrame:
        """Map `dataframe` to private ids, dropping cold users/items (unseen in train).

        Args:
            dataframe (pd.DataFrame): Interactions, with at least 'userId', 'itemId',
                and 'timestamp' columns (and, optionally, 'rating' and 'sessionId').
            is_eval (bool): Whether these are eval interactions.

        Returns:
            pd.DataFrame: The mapped interactions.
        """
        df = dataframe[["userId", "itemId", "timestamp"]].copy()
        df["__u"] = df["userId"].map(self._u_map)
        df["__i"] = df["itemId"].map(self._i_map)
        df["__r"] = dataframe["rating"].to_numpy() if "rating" in dataframe.columns else 1.0
        df["__s"] = dataframe["sessionId"].to_numpy() if "sessionId" in dataframe.columns else 0
        df["__e"] = is_eval
        return df.dropna(subset=["__u", "__i"])

    def get_eval_context(self, row_indices: Any, max_seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Padded context per requested row, plus lengths.

        Args:
            row_indices (Any): Iterable of eval-row indices.
            max_seq_len (int): Maximum context length; longer contexts are
                truncated to their most recent `max_seq_len` items.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Padded context sequences (shape
                `(len(row_indices), max_seq_len)`) and their true lengths.
        """
        seqs, lens = [], []

        # Slice each row's most recent context items off the tape
        for r in row_indices:
            r = int(r)
            cut = int(self._cut[r])
            start = max(int(self._context_start[r]), cut - max_seq_len)
            seqs.append(torch.tensor(self._flat_items[start:cut], dtype=torch.long))
            lens.append(cut - start)

        if not seqs:
            return torch.empty((0, max_seq_len), dtype=torch.long), torch.empty((0,), dtype=torch.long)

        # Pad every context up to the batch's fixed max length
        padded = pad_sequence(seqs, batch_first=True, padding_value=self._n_items)
        return padded, torch.tensor(lens, dtype=torch.long)

    def get_seen_mask(self, row_indices: Any) -> Tuple[np.ndarray, np.ndarray]:
        """(batch row, item) coordinates of the items already seen right before each
        requested row's target: its owner's `seen` items, plus the eval items
        preceding the target.

        Args:
            row_indices (Any): Iterable of eval-row indices.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Batch-relative row and item indices.
        """
        rows = np.asarray(row_indices, dtype=np.int64)

        seen_rows, seen_items = self._seen[self._owner_users[rows]].nonzero()
        prefix_rows, positions = self._expand_ranges(self._prefix_start[rows], self._cut[rows])

        return (
            np.concatenate([seen_rows, prefix_rows]),
            np.concatenate([seen_items, self._flat_items[positions]])
        )

    @cached_property
    def target_items(self) -> List[List[int]]:
        """Private item ids of each row's (deduplicated) target item(s)."""
        return [
            list(dict.fromkeys(self._flat_items[c:e].tolist()))
            for c, e in zip(self._cut, self._target_end)
        ]

    @cached_property
    def target_public_ids(self) -> List[List[Any]]:
        """Public item ids of each row's target item(s)."""
        return [[self._inv_i_map[i] for i in items] for items in self.target_items]

    @cached_property
    def ground_truth(self) -> Dict[Any, Dict[Any, float]]:
        """Public row id -> {public target item id: rating}: each row's ground truth is
        only its own target item(s), never another row's of the same owner."""
        return {
            row_id: {
                self._inv_i_map[int(i)]: float(r)
                for i, r in zip(self._flat_items[c:e], self._flat_ratings[c:e])
            }
            for row_id, c, e in zip(self.row_public_ids, self._cut, self._target_end)
        }

    @cached_property
    def owner_public_ids(self) -> List[Any]:
        """Public user id owning each eval row."""
        return [self._inv_u_map[int(u)] for u in self._owner_users]

    @cached_property
    def row_public_ids(self) -> List[Any]:
        """Public id per eval row: its owner's, or a virtual one when a user can own
        several rows, so `collector.py` doesn't collapse them into a single
        recommendation entry."""
        if not self.multi_row:
            return self.owner_public_ids
        return [f"{u}::{int(k)}" for u, k in zip(self.owner_public_ids, self._local_idx)]

    @cached_property
    def owner_map(self) -> Dict[Any, Any]:
        """Public row id -> public owning-user id, handed to `Evaluator` (through
        `collector.py`) to average metrics within a user's rows before averaging
        across users."""
        return dict(zip(self.row_public_ids, self.owner_public_ids))

    @property
    def owner_users(self) -> np.ndarray:
        """Private (train-space) owning-user index per eval row."""
        return self._owner_users

    @property
    def n_rows(self) -> int:
        """Total number of eval rows."""
        return self._n_rows

    def _expand_ranges(self, starts: np.ndarray, ends: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Expand `[start, end)` ranges into one (range index, position) pair per position.

        Args:
            starts (np.ndarray): Range starts.
            ends (np.ndarray): Range ends (exclusive).

        Returns:
            Tuple[np.ndarray, np.ndarray]: The range index and position of every expanded entry.
        """
        lengths = (ends - starts).astype(np.int64)
        range_idx = np.repeat(np.arange(len(starts)), lengths)
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        return range_idx, np.repeat(starts, lengths) + offsets
