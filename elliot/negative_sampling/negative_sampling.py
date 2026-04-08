from typing import Tuple, List, Optional
import bisect
import random
import numpy as np
from tqdm import tqdm

from elliot.namespace import NegativeSamplingConfig
from elliot.utils.enums import NegativeSamplingStrategy
from elliot.utils.read import Reader
from elliot.utils.write import Writer


def _zero_intervals(n_cols, nnz_sorted):
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
        neg_sampling_config (NegativeSamplingConfig): Configuration object containing negative sampling parameters.
        mappings (Tuple[dict, dict]): User and item mappings from public ids to internal indices.
        inv_mappings (Tuple[List[str], List[str]]): Inverse mappings from internal indices
            to public user and item ids.
        num_users (int): Total number of users.
        num_items (int): Total number of items.
        train_pos_items (List[int]): Positive item indices per user in the training set.
        eval_pos_items (List[int]): Positive item indices per user in the evaluation set.
        evaluation_set (str): Evaluation set name. Defaults to "test".
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
        mappings: Tuple[dict, dict],
        inv_mappings: Tuple[List[str], List[str]],
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

        self.neg_sampling_config = neg_sampling_config
        self.fold_index = fold_index

        self._u_map, self._i_map = mappings
        self._inv_u_map, self._inv_i_map = inv_mappings

        self._num_users = num_users
        self._num_items = num_items

        self.merged_pos_items = self._merge_positives(train_pos_items, eval_pos_items)
        self._evaluation_set = evaluation_set

        np.random.seed(random_seed)
        random.seed(random_seed)

    def _merge_positives(
        self,
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
