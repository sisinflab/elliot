from typing import Tuple, List
from types import SimpleNamespace

import bisect
import random
import numpy as np
from tqdm import tqdm

from elliot.utils.enums import NegativeSamplingStrategy
from elliot.utils.config import NegativeSamplingConfig
from elliot.utils.sparse import zero_intervals
from elliot.utils.read import Reader
from elliot.utils.write import Writer

reader = Reader()
writer = Writer()


class NegativeSampler:
    """The NegativeSampler class is responsible for performing negative sampling in a recommendation system.

    This class generates negative samples for training, validation, and testing,
    using configurable strategies.

    Supported sampling strategies:

    - `random`: Uniformly samples a predefined number of negative items for each user.
    - `fixed`: Uses negative items provided in an external file.

    Args:
        config (SimpleNamespace): Configuration object containing negative sampling parameters.
        mappings (Tuple[dict, dict]): User and item mappings from public ids to internal indices.
        inv_mappings (Tuple[np.ndarray, np.ndarray]): Inverse mappings from internal indices
            to public user and item ids.
        num_users (int): Total number of users.
        num_items (int): Total number of items.
        pos_items (Tuple[List[List[int]], List[List[int]], List[List[int]]]):
            Positive item indices per user for train, validation, and test splits.
        random_seed (int): Random seed for reproducibility; default is 42.

    To configure the negative sampling, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      negative_sampling:
        strategy: random|fixed
        num_negatives: 5
        save_folder: this/is/the/path
        read_folder: this/is/the/path
    """

    config: NegativeSamplingConfig

    def __init__(
        self,
        config: SimpleNamespace,
        mappings: Tuple[dict, dict],
        inv_mappings: Tuple[np.ndarray, np.ndarray],
        num_users: int,
        num_items: int,
        pos_items: Tuple[List[List[int]], List[List[int]], List[List[int]]],
        random_seed: int = 42
    ):
        self.config = NegativeSamplingConfig(**vars(config))
        self.reader_config = self._get_reader_writer_config()
        self.writer_config = self._get_reader_writer_config()

        self._u_map, self._i_map = mappings
        self._inv_u_map, self._inv_i_map = inv_mappings

        self._num_users = num_users
        self._num_items = num_items

        train, val, test = pos_items
        self.merged_pos_items = self._merge_positives(train, val, test)
        self.add_val = len(val) > 0

        np.random.seed(random_seed)
        random.seed(random_seed)

    def _get_reader_writer_config(self):
        return {
            "sep": "\t",
            "ext": ".tsv"
        }

    def _merge_positives(
        self,
        train: List[List[int]],
        val: List[List[int]],
        test: List[List[int]]
    ) -> List[List[int]]:
        """Merge positive interactions across data splits for each user.

        Args:
            train (List[List[int]]): Positive item indices per user in the training split.
            val (List[List[int]]): Positive item indices per user in the validation split.
            test (List[List[int]]): Positive item indices per user in the test split.

        Returns:
            List[List[int]]: List of merged unique positive item indices per user.
        """
        all_items_list = []

        iterables = (train, val, test) if val else (train, test)

        for items in zip(*iterables):
            all_items = set().union(*items)
            all_items_list.append(list(all_items))

        return all_items_list

    def sample(self) -> Tuple[List[List[int]], List[List[int]]]:
        """Generate negative samples for validation and test splits.

        Returns:
            Tuple[List[List[int]], List[List[int]]]: Negative item indices per user
                for validation and test.
        """
        val_negative_items = self.process_sampling(validation=True) if self.add_val else []
        test_negative_items = self.process_sampling(validation=False)

        return val_negative_items, test_negative_items

    def process_sampling(self, validation: bool = False) -> List[List[int]]:
        """Run negative sampling according to the configured strategy.

        Args:
            validation (bool): Whether to generate negatives for validation or test; default is False.

        Returns:
            List[List[int]]: Negative item indices per user.
        """
        if self.config.strategy == NegativeSamplingStrategy.RANDOM:
            neg = self.random_strategy(validation)
        else:
            neg = self.fixed_strategy(validation)

        return neg

    def random_strategy(self, validation: bool = False) -> List[List[int]]:
        """Sample negative items uniformly at random for each user.

        Args:
            validation (bool): Whether to generate negatives for validation or test; default is False.

        Returns:
            List[List[int]]: Randomly sampled negative item indices per user.
        """
        data = self.merged_pos_items
        rows, neg = [], []

        iter_data = tqdm(
            data,
            total=len(data),
            desc=f"Sampling negatives for {"test" if not validation else "validation"}",
            leave=False
        )

        for i, u_indices in enumerate(iter_data):
            # Compute candidates number
            candidate_negatives_count = self._num_items - len(u_indices)

            # Randomly sample negatives...
            if candidate_negatives_count > self.config.num_negatives:
                sampled = self._sample_by_random_uniform(u_indices)
            # ...or pick them all
            else:
                negatives = set(range(self._num_items)) - set(u_indices)
                negatives = sorted(negatives)
                sampled = negatives

            # Append sampled negatives
            neg.append(sampled)

        # Optionally save negatives to file
        if self.config.save_on_disk:
            self._save_to_file(neg, validation=validation)

        return neg

    def _sample_by_random_uniform(self, indices: List[int]) -> List[int]:
        """Uniformly sample negative items excluding positive interactions.

        Args:
            indices (List[int]): Positive item indices for a user.

        Returns:
            List[int]: Randomly sampled negative item indices.
        """
        # Compute empty intervals between item ids (candidate negatives)
        intervals = zero_intervals(self._num_items, indices)

        # Set some initial parameters
        lengths = [b - a + 1 for (a, b) in intervals]
        cum = []
        s = 0
        for L in lengths:
            s += L
            cum.append(s)

        total = cum[-1]
        sampled = set()

        while len(sampled) < self.config.num_negatives:
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

    def _save_to_file(self, neg: List[List[int]], validation: bool = False):
        """Save negative samples to disk using public ids.

        Args:
            neg (List[List[int]]): Negative item indices per user (private ids).
            validation (bool): Whether the negatives belong to validation or test split; default is False.
        """
        neg_dict = {}

        # Build negatives dict with public ids
        for u, items in enumerate(neg):
            user_id = self._inv_u_map[u]
            mapped_items = [self._inv_i_map[i] for i in items]
            neg_dict[user_id] = mapped_items

        # Write to file
        writer.write_negatives(
            neg_dict=neg_dict,
            save_folder=self.config.save_folder,
            scope="val" if validation else "test",
            **self.writer_config
        )

    def fixed_strategy(self, validation: bool = False) -> List[List[int]]:
        """Load precomputed negative samples from disk.

        Args:
            validation (bool): Whether to load validation or test negatives; default is False.

        Returns:
            List[List[int]]: Negative item indices per user mapped to private ids.
        """
        # Read from file
        neg_dict = reader.read_negatives(
            read_folder=self.config.read_folder,
            scope="val" if validation else "test",
            **self.reader_config
        )

        neg = [[]] * self._num_users

        iter_data = tqdm(
            neg_dict.items(),
            desc=f"Loading negatives for {"test" if not validation else "validation"}",
            leave=False
        )

        # Build negatives list with private ids
        for user_id, neg_list in iter_data:
            if user_id not in self._u_map:
                continue

            row = self._u_map[user_id]
            neg[row] = [self._i_map[i] for i in neg_list if i in self._i_map]

        return neg
