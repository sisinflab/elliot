import warnings
import random
import typing as t
import numpy as np
from scipy import sparse as sp
from ast import literal_eval as make_tuple
from tqdm import tqdm

from elliot.utils import sparse

np.random.seed(42)
random.seed(42)


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

    def __init__(self, data):
        """
        Initializes the NegativeSampler object.

        Args:
            data (DataSet): Dataset object containing users, items, ratings, configuration,
                validation, and test sets.

        Raises:
            ValueError: If the specified sampling strategy is unrecognized.
        """
        self.ns = data.config.negative_sampling
        self.public_users = data.public_users
        self.public_items = data.public_items
        self.private_users = data.private_users
        self.private_items = data.private_items
        self.i_train = data.sp_i_train_ratings
        self.batch_size = data.batch_size
        self.n_batches = len(data)
        self.val = data.val_dict
        self.test = data.test_dict

        self.param_ranges = {}
        self._compute_param_ranges()

        strategy = self._get_validated_attr("strategy", str)
        if strategy == "random":
            self._strategy = self.random_strategy
            self._num_items = self._get_validated_attr("num_items")
            self._file_path = self._get_validated_attr("file_path", str, False)
        elif strategy == "fixed":
            self._strategy = self.fixed_strategy
            self._files = self._get_validated_attr("files", str)
        else:
            raise ValueError(f"Unrecognized sampling strategy: '{strategy}'.")

    def sample(self) -> t.Tuple[t.Optional[sp.csr_matrix], t.Optional[sp.csr_matrix]]:
        """
        Generates negative samples for validation and test sets.

        Returns:
            tuple[sp.csr_matrix | None, sp.csr_matrix | None]: A pair of sparse matrices representing
                negative samples for validation and test sets. Each element may be None if
                the corresponding dictionary is not provided.
        """
        val_negative_items = None
        if self.val is not None:
            val_negative_items = self.process_sampling(validation=True)

        test_negative_items = None
        if self.test is not None:
            test_negative_items = self.process_sampling(validation=False)

        return val_negative_items, test_negative_items

    def process_sampling(self, validation: bool = False) -> sp.csr_matrix:
        """
        Performs batch-wise negative sampling for validation or test data.

        Args:
            validation (bool, optional): Whether to process validation data. Defaults to False.

        Returns:
            sp.csr_matrix: Sparse mask containing negative samples combined with positive interactions.
        """
        shape = (len(self.public_users), len(self.private_items))
        i_test = sparse.build_sparse(self.test, shape, self.public_users, self.public_items)
        i_val = sparse.build_sparse(self.val, shape, self.public_users, self.public_items) if validation else None

        rows, cols = [], []
        text_message = f"Performing negative sampling using {"test" if not validation else "validation"} data"

        with tqdm(total=self.n_batches, desc=text_message) as tq:
            for batch_start in range(0, len(self.public_users), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(self.public_users))
                batch_train = self.i_train[batch_start:batch_end]
                batch_test = i_test[batch_start:batch_end]
                batch_candidates = np.where(((batch_train + batch_test).toarray() == 0), True, False)

                batch_rows, batch_cols = self._strategy(batch_candidates, batch_start, validation)
                rows.append(batch_rows)
                cols.append(batch_cols)

                tq.update()

        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        sampled_mask = sparse.build_sparse_mask(rows, cols, shape=self.i_train.shape)
        mask = sampled_mask + (i_val if validation else i_test)
        return mask

    def random_strategy(
        self,
        candidate_negatives: np.ndarray,
        batch_start: int,
        validation: bool
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Randomly samples negative items for each user in the batch.

        Args:
            candidate_negatives (np.ndarray): Boolean array indicating candidate negative items
                (True for eligible negatives).
            batch_start (int): Index of the first user in the current batch.
            validation (bool): Only for compatibility.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of arrays containing rows and cols indices
                of sampled negatives.
        """
        rows, cols = self._sample_by_random_uniform(candidate_negatives)

        rows += batch_start
        self._save_to_file(rows, cols, batch_start)

        return rows, cols

    def fixed_strategy(
        self,
        candidate_negatives: np.ndarray,
        batch_start: int,
        validation: bool
    ) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Loads fixed negative samples from file for the batch.

        Args:
            candidate_negatives (np.ndarray): Only for compatibility.
            batch_start (int): Index of the first user in the current batch.
            validation (bool): Whether the batch belongs to validation data.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of arrays containing rows and cols indices
                of sampled negatives.
        """
        if not isinstance(self._files, list):
            self._files = [self._files]

        file_ = self._files[1] if validation and len(self._files) > 1 else self._files[0]
        return self._read_from_file(file_, batch_start)

    def _sample_by_random_uniform(self, data: np.ndarray) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Randomly samples negative items row by row from the current batch.

        Args:
            data (np.ndarray): Boolean array indicating candidate negative items for each user.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of arrays containing rows and cols indices
                of sampled negatives, referring to the current batch.
        """
        rows, cols = [], []

        for row in range(data.shape[0]):
            candidate_negatives = np.flatnonzero(data[row])

            if len(candidate_negatives) > self._num_items:
                sampled = np.random.choice(candidate_negatives, size=self._num_items, replace=False)
            else:
                sampled = candidate_negatives

            rows.append(np.full(sampled.shape, row, dtype=int))
            cols.append(sampled)

        return np.concatenate(rows), np.concatenate(cols)

    def _save_to_file(self, rows: np.ndarray, cols: np.ndarray, batch_start: int) -> None:
        """
        Saves negative samples of the current batch to file.

        Args:
            rows (np.ndarray): Row indices of sampled negatives.
            cols (np.ndarray): Column indices of sampled negatives.
            batch_start (int): Starting index of the batch (used to append or overwrite file).
        """
        mode = "w" if batch_start == 0 else "a"
        user_to_items = {}
        for r, c in zip(rows, cols):
            user_to_items.setdefault(r, []).append(c)

        with open(self._file_path, mode) as f:
            for u, items in user_to_items.items():
                user_id = self.private_users[u]
                mapped_items = [self.private_items[i] for i in items]
                f.write(f"{(user_id,)}\t" + "\t".join(map(str, mapped_items)) + "\n")

    def _read_from_file(self, file_path: str, batch_start: int) -> t.Tuple[np.ndarray, np.ndarray]:
        """
        Reads negative samples from a file for the current batch.

        Args:
            file_path (str): Path to the file containing fixed negative samples.
            batch_start (int): Index of the first user in the batch.

        Returns:
            tuple[np.ndarray, np.ndarray]: A pair of arrays containing rows and cols indices
                of negative samples.
        """
        rows, cols = [], []
        with open(file_path) as file:
            for idx, line in enumerate(file):
                if idx < batch_start:
                    continue
                if idx >= batch_start + self.batch_size:
                    break

                line = line.rstrip("\n").split('\t')
                user_id = int(make_tuple(line[0])[0])
                if user_id not in self.public_users:
                    continue

                row = self.public_users[user_id]
                for i in line[1:]:
                    item = int(i)
                    if item in self.public_items:
                        col = self.public_items[item]
                        rows.append(row)
                        cols.append(col)

        return np.array(rows), np.array(cols)

    def _get_validated_attr(
        self,
        attr: str,
        expected_type: t.Type = int,
        required: bool = True
    ) -> t.Any:
        """
        Retrieves and validates a single attribute from the configuration namespace.

        This method performs several checks:

        - Ensures the attribute exists if `required` is True.
        - Confirms the value is of the expected type.
        - For numeric values, ensures it is non-negative and within a valid range
          computed from the dataset (`data`), using `_compute_param_ranges` and `_check_interval`.

        Args:
            attr (str): The attribute name to retrieve and validate.
            expected_type (type, optional): The expected data type for the attribute. Defaults to int.
            required (bool, optional): Whether the attribute is required to be present. Defaults to True.

        Returns:
            Any: The validated value of the attribute.

        Raises:
            AttributeError: If the attribute is missing and `required` is True.
            TypeError: If the attribute is not of the expected type.
            ValueError: If the attribute is a numeric value outside valid bounds.
        """
        val = getattr(self.ns, attr, None)
        allowed_type_names = f"'{expected_type.__name__}'"

        if required and val is None:
            raise AttributeError(f"Missing required attribute: '{attr}'.")
        if val is not None and not isinstance(val, expected_type):
            raise TypeError(f"Attribute '{attr}' must be of type {allowed_type_names}, got '{type(val).__name__}'.")

        # Optional value constraints
        if isinstance(val, int):
            self._check_interval(attr, val)

        return val

    def _compute_param_ranges(self) -> None:
        """
        Computes and sets valid parameter ranges for different sampling strategies
        based on the characteristics of the dataset.

        The computed ranges are stored in `self.param_ranges` and include:

        - `num_items`: Minimum and maximum number of negative items to sample per user.

        Raises:
            Warning: Emits a warning if there are too few items for negative sampling.
        """
        min_items = 100
        positives = np.diff(self.i_train.indptr).mean()
        N = self.i_train.shape[1]
        if N < min_items:
            self.val = self.test = None
            warnings.warn(f"Cannot perform negative sampling: "
                          f"at least {min_items} items are required (current: {N}).")
        else:
            self.param_ranges["num_items"] = [
                round(0.00001 * N * positives) * 10,
                round(0.0001 * N * positives) * 10
            ]

    def _check_interval(self, attr: str, val: int) -> None:
        """
        Validates that a numeric attribute value falls within a predefined acceptable range.

        The valid range for the attribute is retrieved from `self.param_ranges[attr]`.

        Args:
            attr (str): The name of the attribute to validate.
            val (int | float): The value to check.

        Raises:
            ValueError: If `val` is outside the valid range for the given attribute.
        """
        if attr not in self.param_ranges:
            return
        min_val, max_val = self.param_ranges[attr]
        if not (min_val <= val <= max_val):
            raise ValueError(f"Attribute '{attr}' must be between {min_val} and {max_val}, "
                             f"based on the provided dataset.")
