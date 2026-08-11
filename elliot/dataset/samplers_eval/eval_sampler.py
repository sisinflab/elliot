from typing import Any, Dict, List, Optional, Tuple
import bisect
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from elliot.namespace import NegativeSamplingConfig
from elliot.utils.enums import NegativeSamplingStrategy
from elliot.utils.read import Reader
from elliot.utils.write import Writer


def _zero_intervals(n_cols: int, nnz_sorted: List[int]) -> List[Tuple[int, int]]:
    """Compute the maximal `[start, end]` column intervals not covered by the
    sorted, already-present column indices `nnz_sorted`, over a row of width `n_cols`.

    Args:
        n_cols (int): Total number of columns in the row.
        nnz_sorted (List[int]): Sorted column indices already present (excluded) in
            the row.

    Returns:
        List[Tuple[int, int]]: The maximal zero (candidate) intervals, in column order.
    """
    intervals = []
    prev = -1
    for c in nnz_sorted:
        if c > prev + 1:
            intervals.append((prev + 1, c - 1))
        prev = c
    if prev < n_cols - 1:
        intervals.append((prev + 1, n_cols - 1))
    return intervals


class NegativeSampler:
    """The NegativeSampler class is responsible for performing negative sampling in a recommendation system.

    This class generates negative samples for training, validation, and testing,
    using configurable strategies.

    Supported sampling strategies:

    - `random`: Uniformly samples a predefined number of negative items for each user.
    - `fixed`: Uses negative items provided in an external file.

    Args:
        neg_sampling_config (NegativeSamplingConfig): Configuration object containing
            negative sampling parameters.
        mappings (Tuple[Dict[Any, int], Dict[Any, int]]): (user, item) mappings from
            public ids to private indices.
        inv_mappings (Tuple[List[Any], List[Any]]): Inverse (user, item) mappings from
            private indices back to public ids.
        num_users (int): Total number of users.
        num_items (int): Total number of items.
        train_pos_items (List[List[int]]): Positive item indices per user in the training set.
        eval_pos_items (List[List[int]]): Positive item indices per user in the evaluation set.
        evaluation_set (str): Name of this fold's eval split ("test" or "validation").
            Defaults to "test".
        fold_index (Tuple[int, Optional[int]]): Tuple containing the complete fold index.
        random_seed (int): Random seed for reproducibility. Defaults to 42.

    To configure the negative sampling, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      negative_sampling:
        strategy: random|fixed
        num_negatives: 5
        save_folder: this/is/the/path
        read_folder: this/is/the/path
        leave_one_out: True|False
    """

    neg_sampling_config: NegativeSamplingConfig

    def __init__(
        self,
        neg_sampling_config: NegativeSamplingConfig,
        mappings: Tuple[Dict[Any, int], Dict[Any, int]],
        inv_mappings: Tuple[List[Any], List[Any]],
        num_users: int,
        num_items: int,
        train_pos_items: List[List[int]],
        eval_pos_items: List[List[int]],
        evaluation_set: str = "test",
        fold_index: Tuple[int, Optional[int]] = (0, None),
        random_seed: int = 42
    ):
        self.reader = Reader()
        self.writer = Writer()

        # Initializing variables
        self.neg_sampling_config = neg_sampling_config
        self.fold_index = fold_index

        self._u_map, self._i_map = mappings
        self._inv_u_map, self._inv_i_map = inv_mappings

        self._num_users = num_users
        self._num_items = num_items

        self._evaluation_set = evaluation_set

        self.merged_pos_items = self._merge_positives(train_pos_items, eval_pos_items)

        np.random.seed(random_seed)
        random.seed(random_seed)

    @staticmethod
    def _merge_positives(
        train_pos: List[List[int]],
        eval_pos: List[List[int]]
    ) -> List[List[int]]:
        """Merge positive interactions across data splits for each user.

        Args:
            train_pos (List[List[int]]): Positive item indices per user in the training set.
            eval_pos (List[List[int]]): Positive item indices per user in the evaluation set.

        Returns:
            List[List[int]]: List of merged unique positive item indices per user.
        """
        all_items_list = []

        for items in zip(train_pos, eval_pos):
            all_items = set().union(*items)
            all_items_list.append(list(all_items))

        return all_items_list

    def sample(self) -> List[List[int]]:
        """Run negative sampling according to the configured strategy.

        Returns:
            List[List[int]]: Negative item indices per user.
        """
        if self.neg_sampling_config.strategy == NegativeSamplingStrategy.RANDOM:
            neg = self.random_strategy()
        else:
            neg = self.fixed_strategy()

        return neg

    def random_strategy(self) -> List[List[int]]:
        """Sample negative items uniformly at random for each user.

        Returns:
            List[List[int]]: Randomly sampled negative item indices per user.
        """
        data = self.merged_pos_items
        rows, neg = [], []

        iter_data = tqdm(
            data,
            total=len(data),
            desc=f"Sampling negatives for {self._evaluation_set}",
            leave=False
        )

        for i, u_indices in enumerate(iter_data):
            # Compute the number of candidates
            candidate_negatives_count = self._num_items - len(u_indices)

            # Randomly sample negatives...
            if candidate_negatives_count > self.neg_sampling_config.num_negatives:
                sampled = self._sample_by_random_uniform(u_indices)
            # ...or pick them all
            else:
                negatives = set(range(self._num_items)) - set(u_indices)
                negatives = sorted(negatives)
                sampled = negatives

            # Append sampled negatives
            neg.append(sampled)

        # Optionally save negatives to file
        if self.neg_sampling_config.save_on_disk:
            self._save_to_file(neg)

        return neg

    def _sample_by_random_uniform(self, indices: List[int]) -> List[int]:
        """Uniformly sample negative items excluding positive interactions.

        Args:
            indices (List[int]): Positive item indices for a user.

        Returns:
            List[int]: Randomly sampled negative item indices.
        """
        # Compute empty intervals between item ids (candidate negatives)
        intervals = _zero_intervals(self._num_items, indices)

        # Set some initial parameters
        lengths = [b - a + 1 for (a, b) in intervals]
        cum = []
        s = 0
        for L in lengths:
            s += L
            cum.append(s)

        total = cum[-1]
        sampled = set()

        while len(sampled) < self.neg_sampling_config.num_negatives:
            # Pick a random sample
            u = random.randrange(total)

            # Find the range which it belongs to
            idx = bisect.bisect_right(cum, u)
            start, end = intervals[idx]
            prev_cum = cum[idx - 1] if idx > 0 else 0

            # Compute the offset within the range
            offset = u - prev_cum

            sampled.add(start + offset)

        return list(sampled)

    def _save_to_file(self, neg: List[List[int]]):
        """Save negative samples to disk using public ids.

        Args:
            neg (List[List[int]]): Negative item indices per user (private ids).
        """
        neg_dict = {}

        # Build negatives dict with public ids
        for u, items in enumerate(neg):
            user_id = self._inv_u_map[u]
            mapped_items = [self._inv_i_map[i] for i in items]
            neg_dict[user_id] = mapped_items

        # Write to file
        self.writer.write_negatives(
            neg_dict=neg_dict,
            save_folder=self.neg_sampling_config.save_folder,
            fold_index=self.fold_index,
            sep=self.neg_sampling_config.writer.sep,
            ext=self.neg_sampling_config.writer.ext,
        )

    def fixed_strategy(self) -> List[List[int]]:
        """Load precomputed negative samples from disk.

        Returns:
            List[List[int]]: Negative item indices per user mapped to private ids.
        """
        # Read from file
        neg_dict = self.reader.read_negatives(
            read_folder=self.neg_sampling_config.read_folder,
            fold_index=self.fold_index,
            sep=self.neg_sampling_config.reader.sep,
            ext=self.neg_sampling_config.reader.ext,
        )

        neg = [[]] * self._num_users

        iter_data = tqdm(
            neg_dict.items(),
            desc=f"Loading negatives for {self._evaluation_set}",
            leave=False
        )

        # Build negatives list with private ids
        for user_id, neg_list in iter_data:
            if user_id not in self._u_map:
                continue

            row = self._u_map[user_id]
            neg[row] = [self._i_map[i] for i in neg_list if i in self._i_map]

        return neg


class NegEvalDataset(Dataset):
    """Evaluation dataset pairing sampled negatives with the ground-truth positive(s)
    for each user, used when `NegativeSamplingConfig` is configured (as opposed to
    `FullEvalDataset`'s full-ranking evaluation).

    Args:
        num_users (int): Total number of users (or eval rows, for SESSION_ONLY
            evaluation) to evaluate.
        eval_neg_items (List[List[int]]): Negative item indices per row, already
            sampled. Under SESSION_ONLY evaluation this is the owning user's single,
            once-sampled negative list broadcast to every session row they own (see
            `DataSet._get_user_negatives()`), so that FLAT and SESSION_ONLY models are
            always scored against the same negatives.
        eval_pos_items (List[List[int]]): Ground-truth positive item indices per user.
        evaluation_set (str): Name of this fold's eval split ("test" or "validation").
            Defaults to "test".
        leave_one_out (bool): If True, only the last ground-truth positive per user is
            kept. Defaults to False.
    """

    def __init__(
        self,
        num_users: int,
        eval_neg_items: List[List[int]],
        eval_pos_items: List[List[int]],
        evaluation_set: str = "test",
        leave_one_out: bool = False
    ):
        # Initializing variables
        self.num_users = num_users
        self.leave_one_out = leave_one_out

        self._evaluation_set = evaluation_set

        self.eval_items = self._add_indices(eval_neg_items, eval_pos_items)

    def _add_indices(self, neg: List[List[int]], pos: List[List[int]]) -> Optional[List[torch.Tensor]]:
        """Add test or validation samples to the sampled negatives.

        Args:
            neg (List[List[int]]): Negative item indices per user.
            pos (List[List[int]]): Ground-truth positive item indices per user.

        Returns:
            Optional[List[torch.Tensor]]: Per-user tensors of negative items followed
                by the ground-truth positive(s), or None if `neg` is empty.
        """
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

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        return index, self.eval_items[index]

    @staticmethod
    def collate_fn(batch: List[Tuple[int, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of (user index, items) pairs, padding the ragged item
        tensors to the batch's longest one.

        Args:
            batch (List[Tuple[int, torch.Tensor]]): The batch of (user index, items)
                pairs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The batched user indices and the
                padded (`-1`-filled) item tensor.
        """
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
    """Full-ranking evaluation dataset: one entry per user, with no sampled
    negatives (candidates are every item, ranked at evaluation time), used when no
    `NegativeSamplingConfig` is configured.

    Args:
        num_users (int): Total number of users (or eval rows, for SESSION_ONLY
            evaluation) to evaluate.
    """

    def __init__(self, num_users: int):
        # Initializing variables
        self.num_users = num_users

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, index: int) -> int:
        return index

    @staticmethod
    def collate_fn(batch: List[int]) -> Tuple[torch.Tensor, None]:
        """Collate a batch of user indices; there is no per-user item tensor to pad.

        Args:
            batch (List[int]): The batch of user indices.

        Returns:
            Tuple[torch.Tensor, None]: The batched user indices, and `None` in place
                of an item tensor.
        """
        return torch.tensor(batch), None
