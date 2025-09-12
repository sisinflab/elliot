import typing as t
import pandas as pd
import numpy as np
import math
import shutil
import os
from types import SimpleNamespace
from elliot.utils.folder import create_folder_by_index


class Splitter:
    """
    The Splitter class is responsible for performing various dataset splitting strategies,
    such as Train/Test, Train/Validation/Test, K-Folds, temporal splits, and random subsampling.
    It supports both in-memory splits and optionally saving the results to disk.

    This class is designed to work with user-item interaction data in a recommender systems context,
    where splitting must often respect user-level chronology.

    Attributes:
        data (pd.DataFrame): The dataset to be split, typically containing at least
            'userId' and 'timestamp' columns.
        splitting_ns (SimpleNamespace): Namespace object containing configuration
            for the desired splitting strategy.
        random_seed (int): Seed for reproducibility of random-based strategies.
        save_on_disk (bool): Indicates whether the generated splits should be saved to disk.
        save_folder (str | None): Directory path to store the splits if saving is enabled.
        param_ranges (dict): Stores dynamically computed min/max ranges for
            configurable attributes based on the input dataset.

    Supported splitting strategies:

    - `fixed_timestamp`: Splits data using a predefined or computed timestamp threshold.
    - `best_timestamp`: Automatically selects the best timestamp across users for splitting.
    - `temporal_hold_out`: Uses the most recent interactions (per user) for testing.
    - `temporal_leave_n_out`: Leaves the last N events per user as the test set.
    - `random_subsampling`: Randomly subsamples training and testing data across folds.
    - `random_cross_validation (K-Folds)`: Performs stratified K-Fold cross-validation.

    To configure the data splitting, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      splitting:
        save_on_disk: True|False
        save_path: "path"
        test_splitting:
          strategy: fixed_timestamp|temporal_hold_out|random_subsampling|random_cross_validation
          timestamp: best|1609786061
          test_ratio: 0.2
          leave_n_out: 1
          folds: 5
        validation_splitting:
          strategy: fixed_timestamp|temporal_hold_out|random_subsampling|random_cross_validation
          timestamp: best|1609786061
          test_ratio: 0.2
          leave_n_out: 1
          folds: 5

    Notes:
        Splitting is required and will be applied only if `data_config.strategy` is set to 'dataset'.
    """

    def __init__(self, data: pd.DataFrame, splitting_ns: SimpleNamespace, random_seed: int = 42):
        """
        Initializes the Splitter object to manage dataset splitting strategies.

        Args:
            data (pd.DataFrame): The dataset to be split.
            splitting_ns (SimpleNamespace): A namespace object containing splitting configuration.
            random_seed (int, optional): Random seed for reproducibility. Defaults to 42.
        """
        self.random_seed = random_seed
        self.data = data
        self.splitting_ns = splitting_ns
        self.save_on_disk = False
        self.save_folder = None

        self.param_ranges = {}

    def process_splitting(
        self
    ) -> t.List[t.Tuple[t.Union[pd.DataFrame, t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]], pd.DataFrame]]:
        """
        Executes the configured splitting strategy (Train/Test or Train/Validation/Test).

        Returns:
            list[tuple[pd.DataFrame | list[tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]:
                A list of (train, test) or ((train, val), test) tuples.
        """
        np.random.seed(self.random_seed)
        data = self.data
        splitting_ns = self.splitting_ns

        if getattr(splitting_ns, "save_on_disk", False):
            self.save_on_disk = True
            self.save_folder = splitting_ns.save_folder

        # [(train_0,test_0), (train_1,test_1), (train_2,test_2), (train_3,test_3), (train_4,test_4)]
        tuple_list = self.handle_hierarchy(data, splitting_ns.test_splitting)

        if hasattr(splitting_ns, "validation_splitting"):
            tuple_list = [
                (self.handle_hierarchy(train, splitting_ns.validation_splitting), test)
                for train, test in tuple_list
            ]
            print("\nRealized a Train/Validation Test splitting strategy\n")
        else:
            print("\nRealized a Train/Test splitting strategy\n")

        if self.save_on_disk:
            if os.path.exists(self.save_folder):
                shutil.rmtree(self.save_folder, ignore_errors=True)
            os.makedirs(self.save_folder)
            self.store_splitting(tuple_list)

        return tuple_list

    def store_splitting(
        self,
        tuple_list: t.List[t.Tuple[t.Union[pd.DataFrame, t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]], pd.DataFrame]]
    ) -> None:
        """
        Saves the generated splits to disk as TSV files if enabled.

        Args:
            tuple_list (list[tuple[pd.DataFrame | list[tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]):
                A list of split tuples to be saved on disk.
        """
        for i, (train_val, test) in enumerate(tuple_list):
            actual_test_folder = create_folder_by_index(self.save_folder, str(i))
            test.to_csv(os.path.abspath(os.sep.join([actual_test_folder, "test.tsv"])), sep='\t', index=False, header=False)
            if isinstance(train_val, list):
                for j, (train, val) in enumerate(train_val):
                    actual_val_folder = create_folder_by_index(actual_test_folder, str(j))
                    val.to_csv(os.path.abspath(os.sep.join([actual_val_folder, "val.tsv"])), sep='\t', index=False, header=False)
                    train.to_csv(os.path.abspath(os.sep.join([actual_val_folder, "train.tsv"])), sep='\t', index=False, header=False)
            else:
                train_val.to_csv(os.path.abspath(os.sep.join([actual_test_folder, "train.tsv"])), sep='\t', index=False, header=False)

    def handle_hierarchy(
        self,
        data: pd.DataFrame,
        valtest_splitting_ns: SimpleNamespace
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Handles the splitting logic based on the selected strategy.

        Args:
            data (pd.DataFrame): The dataset to be split.
            valtest_splitting_ns (SimpleNamespace): Namespace containing strategy parameters.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples.

        Raises:
            ValueError: If the strategy is invalid.
        """
        data = data.reset_index(drop=True)
        self._compute_param_ranges(data)

        strategy = valtest_splitting_ns.strategy

        if strategy == "fixed_timestamp":
            timestamp = self._get_validated_attr(valtest_splitting_ns, "timestamp", (int, str))

            if isinstance(timestamp, int):
                tuple_list = self.splitting_passed_timestamp(data, timestamp)
            else:
                kwargs = {}
                min_below, min_over = self._get_validated_attrs(
                    valtest_splitting_ns, ["min_below", "min_over"], at_least_one=False
                )
                if min_below is not None:
                    kwargs["min_below"] = min_below
                if min_over is not None:
                    kwargs["min_over"] = min_over
                tuple_list = self.splitting_best_timestamp(data, **kwargs)

        elif strategy == "temporal_hold_out":
            test_ratio, leave_n_out = self._get_validated_attrs(
                valtest_splitting_ns, ["test_ratio", "leave_n_out"], [float, int]
            )

            if test_ratio is not None:
                tuple_list = self.splitting_temporal_holdout(data, test_ratio)
            else:
                tuple_list = self.splitting_temporal_leave_n_out(data, leave_n_out)

        elif strategy == "random_subsampling":
            folds = self._get_validated_attr(valtest_splitting_ns, "folds")

            test_ratio, leave_n_out = self._get_validated_attrs(
                valtest_splitting_ns, ["test_ratio", "leave_n_out"], [float, int]
            )

            if test_ratio is not None:
                tuple_list = self.splitting_random_subsampling_k_folds(data, folds, test_ratio)
            else:
                tuple_list = self.splitting_random_subsampling_k_folds_leave_n_out(data, folds, leave_n_out)

        elif strategy == "random_cross_validation":
            folds = self._get_validated_attr(valtest_splitting_ns, "folds")
            tuple_list = self.splitting_k_folds(data, int(folds))

        else:
            raise ValueError(f"Unrecognized splitting strategy: '{strategy}'.")

        return tuple_list

    def splitting_temporal_holdout(
        self,
        d: pd.DataFrame,
        ratio: float
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the dataset using temporal hold-out strategy.

        Args:
            d (pd.DataFrame): The dataset to split.
            ratio (float): Ratio of data to assign to the test set.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.
        """
        user_size = d.groupby('userId').size()
        user_threshold = np.floor(user_size * (1 - ratio)).astype(int)

        rank = d.groupby('userId')['timestamp'].rank(method='first', ascending=True)
        mask = rank > d['userId'].map(user_threshold)

        return self._split_with_mask(d, mask)

    def splitting_temporal_leave_n_out(
        self,
        d: pd.DataFrame,
        n: int
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the dataset by leaving the last n items (per user) as test data.

        Args:
            d (pd.DataFrame): The dataset to split.
            n (int): Number of items per user to leave out for testing.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.
        """
        rank = d.groupby('userId')['timestamp'].rank(method='first', ascending=False)
        mask = rank <= n

        return self._split_with_mask(d, mask)

    def splitting_passed_timestamp(
        self,
        d: pd.DataFrame,
        timestamp: int
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the dataset based on a fixed timestamp threshold.

        Args:
            d (pd.DataFrame): The dataset to split.
            timestamp (int): Timestamp threshold to separate train and test sets.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.
        """
        mask = d['timestamp'] >= timestamp

        return self._split_with_mask(d, mask)

    def splitting_best_timestamp(
        self,
        d: pd.DataFrame,
        min_below: int = 1,
        min_over: int = 1
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the dataset based on the best timestamp, i.e., the one that maximizes user coverage.

        Args:
            d (pd.DataFrame): The dataset to split.
            min_below (int): Minimum number of interactions before the timestamp. Defaults to 1.
            min_over (int): Minimum number of interactions after the timestamp. Defaults to 1.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.

        Raises:
            ValueError: If no valid timestamp is found for the split.
        """
        all_timestamps = np.sort(d["timestamp"].unique())
        counter_array = np.zeros(len(all_timestamps), dtype=np.uint32)

        user_groups = d.groupby("userId")

        for _, group in user_groups:
            user_ts = np.sort(group["timestamp"].to_numpy())
            n = len(user_ts)
            if n < (min_below + min_over + 1):
                continue
            start = user_ts[min_below]
            end = user_ts[n - min_over - 1]
            # Extract all the timestamps within the interval of the current user...
            start_idx = np.searchsorted(all_timestamps, start, side="left")
            end_idx = np.searchsorted(all_timestamps, end, side="right")
            # ...and update their counters
            counter_array[start_idx:end_idx] += 1

        if counter_array.sum() == 0:
            raise ValueError("No valid timestamp found. Try lowering 'min_below' or 'min_over'.")

        max_votes = np.max(counter_array)
        max_indices = np.where(counter_array == max_votes)[0]
        best_ts = all_timestamps[max_indices[-1]]

        print(f"Best Timestamp: {best_ts}")
        return self.splitting_passed_timestamp(d, int(best_ts))

    def splitting_k_folds(
        self,
        d: pd.DataFrame,
        folds: int
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Performs K-Fold cross-validation splitting.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples, one per fold.
        """
        def k_fold_list_generator(length):
            return np.arange(length) % folds

        return self._split_k_folds(d, k_fold_list_generator, folds)

    def splitting_random_subsampling_k_folds(
        self,
        d: pd.DataFrame,
        folds: int,
        ratio: float
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Performs random subsampling across multiple folds based on a test ratio.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds. Defaults to 5.
            ratio (float): Proportion of data per fold to use as test set.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples, one per fold.
        """
        def subsampling_list_generator(length: int) -> t.List[bool]:
            train = int(math.floor(length * (1 - ratio)))
            test = length - train
            list_ = [False] * train + [True] * test
            np.random.shuffle(list_)
            return list_

        return self._split_k_folds(d, subsampling_list_generator, folds, random_subsampling=True)

    def splitting_random_subsampling_k_folds_leave_n_out(
        self,
        d: pd.DataFrame,
        folds: int,
        n: int
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Performs random leave-n-out subsampling across multiple folds.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds.
            n (int): Number of interactions per user to leave out for testing.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples, one per fold.
        """
        def subsampling_leave_n_out_list_generator(length: int) -> t.List[bool]:
            test = n
            train = length - test
            list_ = [False] * train + [True] * test
            np.random.shuffle(list_)
            return list_

        return self._split_k_folds(d, subsampling_leave_n_out_list_generator, folds, random_subsampling=True)

    def _split_k_folds(
        self,
        data: pd.DataFrame,
        generator: t.Callable,
        folds: int,
        random_subsampling: bool = False,
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits the dataset into K folds for cross-validation.

        This method performs K-Fold splitting grouped by `userId`. For each group (i.e., user),
        it applies a generator function to assign entries to one of the K folds. It supports
        two modes of splitting:

        - **Standard K-Fold:** If `random_subsampling` is False, the generator should return an array
          of integers assigning each row to a fold index in `[0, folds-1]`.
        - **Random Subsampling:** If `random_subsampling` is True, the generator should return a boolean
          mask selecting entries for the current fold.

        Args:
            data (pd.DataFrame): The dataset to split.
            generator (callable): A function that generates either an array of fold assignments or a test mask.
            folds (int): Number of folds.
            random_subsampling (bool): If True, enables random subsampling mode. Defaults to False.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples.

        Raises:
            ValueError: If `folds` or `generator` is None, or if required parameters are missing
                for the selected mode.
        """
        fold_indices = [[] for _ in range(folds)]
        user_groups = data.groupby('userId')

        if random_subsampling:
            for _, group in user_groups:
                idx = group.index.to_numpy()
                for fold_id in range(folds):
                    mask = generator(len(group))
                    fold_indices[fold_id].extend(idx[mask])
        else:
            for _, group in user_groups:
                idx = group.index.to_numpy()
                fold_ids = generator(len(group))
                for fold_id in range(folds):
                    fold_indices[fold_id].extend(idx[fold_ids == fold_id])

        mask_list = []
        for test_idx in fold_indices:
            mask = data.index.isin(test_idx)
            mask_list.append(mask)

        return self._split_with_mask(data, mask_list)

    def _split_with_mask(
        self,
        data: t.Union[pd.DataFrame, t.List[pd.DataFrame]],
        mask: t.Union[pd.Series, t.List[pd.Series]],
    ) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Splits a dataset (or list of datasets) into train and test sets
        based on a boolean mask (or list of masks).

        For each mask provided, it selects the rows where the mask is True as the test set,
        and the complement as the train set. If multiple masks are provided, a corresponding
        train/test pair is generated for each.

        Args:
            data (pd.DataFrame | list[pd.DataFrame]): The dataset(s) to be split.
            mask (pd.Series | list[pd.Series]): Boolean mask(s) indicating test samples.

        Returns:
            list[tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples.

        Raises:
            ValueError: If the test set size exceeds the limit defined by the `param_ranges['test_ratio']` parameter.

        Notes:
            - If a single DataFrame and a single mask are provided, the result will be a single (train, test) tuple.
            - If multiple masks are provided, `data` is assumed to be a single DataFrame reused across splits.
        """
        tuple_list = []
        if not isinstance(mask, list):
            mask = [mask]
        for m in mask:
            ratio = round(m.sum() / m.size, 1)
            if ratio < 0.1 or ratio > 0.4:
                ranges_list = "".join(f"- '{k}': {v}\n" for k, v in self.param_ranges.items())
                raise ValueError(f"Train or test set too reduced. Parameter ranges allowed:\n{ranges_list}"
                                 f"If you are using the strategy with timestamp 'best', "
                                 f"try reducing the parameters 'min_below' or 'min_over'.")
            test = data[m].reset_index(drop=True)
            train = data[~m].reset_index(drop=True)
            tuple_list.append((train, test))
        return tuple_list

    def _get_validated_attr(
        self,
        ns: SimpleNamespace,
        attr: str,
        expected_type: t.Union[t.Type, t.Tuple[t.Type, ...]] = int,
        required: bool = True
    ) -> t.Any:
        """
        Retrieves and validates a single attribute from a configuration namespace.

        This method performs several checks:

        - Ensures the attribute exists if `required` is True.
        - Confirms the value is of the expected type(s).
        - For numeric values, ensures it is non-negative and within a valid range
          computed from the dataset (`data`), using `_compute_param_ranges` and `_check_interval`.
        - For string values, only allows 'best' as a valid input.

        Args:
            ns (SimpleNamespace): The namespace containing configuration parameters.
            attr (str): The attribute name to retrieve and validate.
            expected_type (type | tuple[type, ...], optional): The expected data type(s) for the attribute. Defaults to int.
            required (bool, optional): Whether the attribute is required to be present. Defaults to True.

        Returns:
            Any: The validated value of the attribute.

        Raises:
            AttributeError: If the attribute is missing and `required` is True.
            TypeError: If the attribute is not of the expected type.
            ValueError: If the attribute is a numeric value outside valid bounds,
                        or a string different from "best".
        """
        val = getattr(ns, attr, None)
        allowed_type_names = (
            [exp_t.__name__ for exp_t in expected_type]
            if isinstance(expected_type, tuple)
            else f"'{expected_type.__name__}'"
        )

        if required and val is None:
            raise AttributeError(f"Missing required attribute: '{attr}'.")
        if val is not None and not isinstance(val, expected_type):
            raise TypeError(f"Attribute '{attr}' must be of type {allowed_type_names}, got '{type(val).__name__}'.")

        # Optional value constraints
        if isinstance(val, (int, float)):
            if val < 0:
                raise ValueError(f"Attribute '{attr}' must be non-negative.")
            else:
                self._check_interval(attr, val)
        if isinstance(val, str) and val != 'best':
            raise ValueError(f"Attribute '{attr}' must be 'best' if type 'str' is used.")

        return val

    def _get_validated_attrs(
        self,
        ns: SimpleNamespace,
        attrs: t.List[str],
        expected_types: t.Union[t.Type, t.List[t.Type]] = int,
        at_least_one: bool = True
    ) -> t.Tuple[t.Any, ...]:
        """
        Retrieves and validates multiple attributes from a configuration namespace.

        This method performs bulk validation for a list of attributes, including:

        - Type checking for each attribute against a provided expected type or list of types.
        - Optional presence checking: raises an error if all attributes are missing
          and `at_least_one` is set to True.
        - For numeric values, checks for non-negativity and validity within
          computed ranges using `_compute_param_ranges` and `_check_interval`.

        Args:
            ns (SimpleNamespace): Configuration object containing the attributes to validate.
            attrs (list[str]): List of attribute names to retrieve and validate.
            expected_types (type | list[type], optional): Single type or list of types corresponding
                to each attribute. Defaults to int.
            at_least_one (bool, optional): Whether at least one attribute must be present. Defaults to True.

        Returns:
            tuple[Any, ...]: A tuple of validated attribute values, with `None` for any missing optional attributes.

        Raises:
            AttributeError: If all requested attributes are missing and `at_least_one=True`.
            TypeError: If any present attribute is not of the expected type.
            ValueError: For numeric or string values not satisfying custom constraints.
        """
        if isinstance(expected_types, type):
            expected_types = [expected_types] * len(attrs)

        values = tuple(
            self._get_validated_attr(ns, attr, exp_type, required=False)
            for attr, exp_type in zip(attrs, expected_types)
        )

        if at_least_one and all(v is None for v in values):
            raise AttributeError(f"Missing required attribute: at least one among {attrs} must be provided.")

        return values

    def _compute_param_ranges(self, df: pd.DataFrame) -> None:
        """
        Computes and sets valid parameter ranges for different dataset splitting strategies
        based on the characteristics of the input data.

        The computed ranges are stored in `self.param_ranges` and include:

        - `folds`: Minimum and maximum number of folds allowed.
        - `test_ratio`: Acceptable test size ratio per user.
        - `leave_n_out`: Valid range for the number of interactions to leave out per user.
        - `min_below` and `min_over`: Minimum number of interactions before and after a timestamp for "best timestamp" strategy.
        - `timestamp`: Acceptable timestamp range for fixed timestamp splitting.

        Args:
            df (pd.DataFrame): The dataset containing at least 'userId' and 'timestamp' columns.
        """
        TEST_RATIO_RANGE = [0.1, 0.4]
        user_count = df['userId'].nunique()
        interaction_count = len(df)

        self.param_ranges['folds'] = [3, 10]

        self.param_ranges['test_ratio'] = TEST_RATIO_RANGE

        self.param_ranges['leave_n_out'] = [
            math.ceil(TEST_RATIO_RANGE[0] * interaction_count / user_count),
            math.floor(TEST_RATIO_RANGE[1] * interaction_count / user_count),
        ]

        df_sorted = df.sort_values('timestamp')
        lower_idx = int(interaction_count * (1 - TEST_RATIO_RANGE[1]))
        upper_idx = int(interaction_count * (1 - TEST_RATIO_RANGE[0]))
        min_timestamp = df_sorted.iloc[lower_idx]['timestamp']
        max_timestamp = df_sorted.iloc[upper_idx]['timestamp']
        self.param_ranges['timestamp'] = [
            math.ceil(min_timestamp),
            math.floor(max_timestamp)
        ]

    def _check_interval(self, attr: str, val: t.Union[int, float]) -> None:
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
