from typing import Tuple, Optional, List, Type, Any, Callable
from types import SimpleNamespace

import bisect
import random
import numpy as np

from ast import literal_eval as make_tuple
from tqdm import tqdm

from elliot.utils.enums import NegativeSamplingStrategy
from elliot.utils.sparse import zero_intervals
from elliot.utils.validation import NegativeSamplingValidator


class NegativeSampler:
    """
    The NegativeSampler class is responsible for performing negative sampling in a recommendation system.

    This class generates negative samples for training, validation, and testing,
    using configurable strategies.

    Supported sampling strategies:

    - `random`: Uniformly samples a predefined number of negative items for each user.
    - `fixed`: Uses negative items provided in an external file.

    Attributes:
        ns (SimpleNamespace): Configuration object containing negative sampling settings.
        public_users (list): Mapping of public user IDs.
        public_items (dict): Mapping of public item IDs.
        private_users (list): Mapping of private user IDs.
        private_items (dict): Mapping of private item IDs.
        i_train (sp.csr_matrix): Sparse matrix of training interactions.
        batch_size (int): Size of each batch for sampling.
        n_batches (int): Total number of batches.
        val (dict): Validation data dictionary.
        test (dict): Test data dictionary.
        param_ranges (dict): Valid ranges for configuration parameters.
        _strategy (callable): Sampling strategy function selected based on configuration.
        _num_items (int): Number of negative items to sample per user.
        _file_path (str): File path for saving/loading fixed samples.
        _files (list): List of file paths for fixed sampling.

    To configure the negative sampling, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      negative_sampling:
        strategy: random|fixed
        num_items: 5
        files: [ path/to/file ]
    """

    strategy: NegativeSamplingStrategy
    num_negatives: Optional[int] = 99
    save_on_disk: bool = False
    file_path: Optional[str] = None
    test_file_path: Optional[str] = None
    val_file_path: Optional[str] = None

    def __init__(
        self,
        namespace: SimpleNamespace,
        mappings: Tuple[dict, dict],
        inv_mappings: Tuple[np.ndarray, np.ndarray],
        pos_items: list,
        add_validation_sampling: bool = False,
        random_seed: int = 42
    ):
        self.namespace = namespace

        self.public_users, self.public_items = mappings
        self.private_users, self.private_items = inv_mappings

        self._num_users = len(self.public_users)
        self._num_items = len(self.public_items)

        self.pos_items = pos_items
        self.val = add_validation_sampling

        self.set_params()

        np.random.seed(random_seed)
        random.seed(random_seed)

    def set_params(self):
        """Validate and set object parameters."""
        validator = NegativeSamplingValidator(**vars(self.namespace))

        for name, val in validator.get_validated_params().items():
            setattr(self, name, val)

    def sample(self) -> Tuple[Optional[list], list]:
        val_negative_items = self.process_sampling(validation=True) if self.val else None
        test_negative_items = self.process_sampling(validation=False)

        return val_negative_items, test_negative_items

    def process_sampling(self, validation: bool = False) -> list:
        if self.strategy == NegativeSamplingStrategy.RANDOM:
            neg = self.random_strategy(validation)
        else:
            neg = self.fixed_strategy(validation)

        return neg

    def random_strategy(
        self,
        validation: bool
    ) -> List:
        data = self.pos_items
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
            if candidate_negatives_count > self.num_negatives:
                sampled = self._sample_by_random_uniform(u_indices)
            # ...or pick them all
            else:
                sampled = u_indices

            # Append row and col indices of sampled negatives
            neg.append(sampled)

        # Optionally save negatives to file
        if self.save_on_disk:
            self._save_to_file(neg)

        return neg

    def _sample_by_random_uniform(self, indices) -> list:
        intervals = zero_intervals(self._num_items, indices)

        lengths = [b - a + 1 for (a, b) in intervals]
        cum = []
        s = 0
        for L in lengths:
            s += L
            cum.append(s)

        total = cum[-1]
        sampled = []

        for _ in range(self.num_negatives):
            # Pick a random sample
            u = random.randrange(total)

            # Find the range which it belongs to
            idx = bisect.bisect_right(cum, u)
            start, end = intervals[idx]
            prev_cum = cum[idx - 1] if idx > 0 else 0

            # Compute the offset within the range
            offset = u - prev_cum

            sampled.append(start + offset)

        return sampled

    def _save_to_file(self, neg_per_user) -> None:
        with open(self.file_path, "w") as f:
            for u, items in enumerate(neg_per_user):
                user_id = self.private_users[u]
                mapped_items = [self.private_items[i] for i in items]
                f.write(f"{(user_id,)}\t" + "\t".join(map(str, mapped_items)) + "\n")

    def fixed_strategy(self, validation: bool) -> list:
        file_path = self.val_file_path if validation else self.test_file_path
        neg = [None] * self._num_users

        with open(file_path) as file:
            iter_data = tqdm(
                enumerate(file),
                desc=f"Reading negatives from file for {"test" if not validation else "validation"}",
                leave=False
            )

            for idx, line in iter_data:

                line = line.rstrip("\n").split('\t')
                user_id = str(make_tuple(line[0])[0])
                if user_id not in self.public_users:
                    continue

                row = self.public_users[user_id]
                neg[row] = [self.public_items[i] for i in line[1:]]

        return neg
