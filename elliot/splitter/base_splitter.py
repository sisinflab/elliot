from typing import List, Tuple, Union, Callable
from types import SimpleNamespace
import pandas as pd
import numpy as np
import math

from elliot.utils.enums import SplittingStrategy
from elliot.utils.config import SplittingConfig, check_range, SplittingSingleConfig
from elliot.utils.write import Writer
from elliot.utils import logging as elog

writer = Writer()


class Splitter:
    """The Splitter class is responsible for performing various dataset splitting strategies,
    such as Train/Test, Train/Validation/Test, K-Folds, temporal splits, and random subsampling.
    It supports both in-memory splits and optionally saving the results to disk.

    This class is designed to work with user-item interaction data in a recommender systems context,
    where splitting must often respect user-level chronology.

    Args:
        data (pd.DataFrame): The dataset to be split, typically containing at least
            'userId' and 'timestamp' columns.
        config (SimpleNamespace): Namespace object containing configuration
            for the desired splitting strategy.
        random_seed (int): Random seed, for reproducibility; default is 42.

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
        save_folder: path/to/folder
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

    config: SplittingConfig

    def __init__(self, data: pd.DataFrame, config: SimpleNamespace, random_seed: int = 42):
        self.logger = elog.get_logger(self.__class__.__name__, seed=random_seed)
        writer.logger = self.logger

        self.data = data

        # TODO: remove this by completely changing namespace handling (using pydantic)
        if hasattr(config, "test_splitting"):
            config.test_splitting = vars(config.test_splitting)
        if hasattr(config, "validation_splitting"):
            config.validation_splitting = vars(config.validation_splitting)

        self.config = SplittingConfig(**vars(config))
        self.writer_config = self._get_writer_config()

        np.random.seed(random_seed)

    def _get_writer_config(self):
        return {
            "sep": "\t",
            "ext": ".tsv"
        }

    def process_splitting(
        self
    ) -> List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]:
        """Execute the configured splitting strategy (Train/Test or Train/Validation/Test).

        Returns:
            List[Tuple[Union[pd.DataFrame, List[Tuple[pd.DataFrame, pd.DataFrame]]], pd.DataFrame]]:
                A list of (train, test) or ((train, val), test) tuples.
        """

        tuple_list = self.handle_hierarchy(self.data, self.config.test_splitting)

        if self.config.validation_splitting is not None:
            tuple_list = [
                (self.handle_hierarchy(train, self.config.validation_splitting), test)
                for train, test in tuple_list
            ]
            self.logger.info(
                "Completed data split",
                extra={"context": {"strategy": "train_val_test", "folds": len(tuple_list)}}
            )
        else:
            tuple_list = [([(train, None)], test) for train, test in tuple_list]
            self.logger.info(
                "Completed data split",
                extra={"context": {"strategy": "train_test", "folds": len(tuple_list)}}
            )

        if self.config.save_on_disk:
            writer.write_split(
                fold_dataset=tuple_list,
                save_folder=self.config.save_folder,
                **self.writer_config
            )

        return tuple_list

    def handle_hierarchy(
        self,
        data: pd.DataFrame,
        cfg: SplittingSingleConfig,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Handle the splitting logic based on the selected strategy.

        Args:
            data (pd.DataFrame): The dataset to be split.
            cfg (SplittingSingleConfig): Object containing the configuration for the current splitting.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples.
        """
        data = data.reset_index(drop=True)
        tuple_list = []

        match cfg.strategy:
            case SplittingStrategy.FIXED_TS:
                if cfg.timestamp is not None:
                    self._check_timestamp_range(data, cfg.timestamp)
                    tuple_list = self.splitting_passed_timestamp(data, cfg.timestamp)
                else:
                    tuple_list = self.splitting_best_timestamp(
                        data,
                        cfg.min_below,
                        cfg.min_over
                    )

            case SplittingStrategy.TEMP_HOLDOUT:
                if cfg.test_ratio is not None:
                    tuple_list = self.splitting_temporal_holdout(
                        data,
                        cfg.test_ratio
                    )
                else:
                    self._check_leave_n_out_range(data, cfg.leave_n_out)
                    tuple_list = self.splitting_temporal_leave_n_out(
                        data,
                        cfg.leave_n_out
                    )

            case SplittingStrategy.RAND_SUB_SMP:
                if cfg.test_ratio is not None:
                    tuple_list = self.splitting_random_subsampling_k_folds(
                        data,
                        cfg.folds,
                        cfg.test_ratio
                    )
                else:
                    self._check_leave_n_out_range(data, cfg.leave_n_out)
                    tuple_list = self.splitting_random_subsampling_k_folds_leave_n_out(
                        data,
                        cfg.folds,
                        cfg.leave_n_out
                    )

            case SplittingStrategy.RAND_CV:
                tuple_list = self.splitting_k_folds(data, cfg.folds)

        return tuple_list

    def splitting_temporal_holdout(
        self,
        d: pd.DataFrame,
        ratio: float
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split the dataset using temporal hold-out strategy.

        Args:
            d (pd.DataFrame): The dataset to split.
            ratio (float): Ratio of data to assign to the test set.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.
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
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split the dataset by leaving the last n items (per user) as test data.

        Args:
            d (pd.DataFrame): The dataset to split.
            n (int): Number of items per user to leave out for testing.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.
        """
        rank = d.groupby('userId')['timestamp'].rank(method='first', ascending=False)
        mask = rank <= n

        return self._split_with_mask(d, mask)

    def splitting_passed_timestamp(
        self,
        d: pd.DataFrame,
        timestamp: float
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split the dataset based on a fixed timestamp threshold.

        Args:
            d (pd.DataFrame): The dataset to split.
            timestamp (float): Timestamp threshold to separate train and test sets.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.
        """
        mask = d['timestamp'] >= timestamp

        return self._split_with_mask(d, mask)

    def splitting_best_timestamp(
        self,
        d: pd.DataFrame,
        min_below: int,
        min_over: int
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split the dataset based on the best timestamp, i.e., the one that maximizes user coverage.

        Args:
            d (pd.DataFrame): The dataset to split.
            min_below (int): Minimum number of interactions before the timestamp. Defaults to 1.
            min_over (int): Minimum number of interactions after the timestamp. Defaults to 1.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list containing one (train, test) tuple.

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

        self.logger.info(
            "Selected best timestamp",
            extra={"context": {"best_timestamp": float(best_ts), "min_below": min_below, "min_over": min_over}}
        )
        return self.splitting_passed_timestamp(d, best_ts)

    def splitting_k_folds(
        self,
        d: pd.DataFrame,
        folds: int
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Perform K-Fold cross-validation splitting.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples, one per fold.
        """
        def k_fold_list_generator(length):
            return np.arange(length) % folds

        return self._split_k_folds(d, k_fold_list_generator, folds)

    def splitting_random_subsampling_k_folds(
        self,
        d: pd.DataFrame,
        folds: int,
        ratio: float
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Perform random subsampling across multiple folds based on a test ratio.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds. Defaults to 5.
            ratio (float): Proportion of data per fold to use as test set.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples, one per fold.
        """
        def subsampling_list_generator(length: int) -> List[bool]:
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
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Perform random leave-n-out subsampling across multiple folds.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds.
            n (int): Number of interactions per user to leave out for testing.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples, one per fold.
        """
        def subsampling_leave_n_out_list_generator(length: int) -> List[bool]:
            test = n
            train = length - test
            list_ = [False] * train + [True] * test
            np.random.shuffle(list_)
            return list_

        return self._split_k_folds(d, subsampling_leave_n_out_list_generator, folds, random_subsampling=True)

    def _split_k_folds(
        self,
        data: pd.DataFrame,
        generator: Callable,
        folds: int,
        random_subsampling: bool = False,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split the dataset into K folds for cross-validation.

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
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples.

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

    @staticmethod
    def _split_with_mask(
        data: Union[pd.DataFrame, List[pd.DataFrame]],
        mask: Union[pd.Series, List[pd.Series]],
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """Split a dataset (or list of datasets) into train and test sets
        based on a boolean mask (or list of masks).

        For each mask provided, it selects the rows where the mask is True as the test set,
        and the complement as the train set. If multiple masks are provided, a corresponding
        train/test pair is generated for each.

        Args:
            data (pd.DataFrame | list[pd.DataFrame]): The dataset(s) to be split.
            mask (pd.Series | list[pd.Series]): Boolean mask(s) indicating test samples.

        Returns:
            List[Tuple[pd.DataFrame, pd.DataFrame]]: A list of (train, test) tuples.

        Notes:
            - If a single DataFrame and a single mask are provided, the result will be a single (train, test) tuple.
            - If multiple masks are provided, `data` is assumed to be a single DataFrame reused across splits.
        """
        tuple_list = []
        if not isinstance(mask, list):
            mask = [mask]
        for m in mask:
            test = data[m].reset_index(drop=True)
            train = data[~m].reset_index(drop=True)
            tuple_list.append((train, test))
        return tuple_list

    @staticmethod
    def _check_timestamp_range(df: pd.DataFrame, timestamp: float):
        """Validate that `timestamp` falls within the 10th to 90th percentile of the dataset.

        Args:
            df (pd.DataFrame): DataFrame containing a 'timestamp' column.
            timestamp (float): Timestamp value to validate.

        Raises:
            ValueError: If `timestamp` is outside the computed range.
        """
        interaction_count = len(df)

        if 'timestamp' in list(df.columns):
            df_sorted = df.sort_values('timestamp')

            lower_idx = int(interaction_count * 0.1)
            upper_idx = int(interaction_count * 0.9)

            min_timestamp = df_sorted.iloc[lower_idx]['timestamp']
            max_timestamp = df_sorted.iloc[upper_idx]['timestamp']

            check_range('timestamp', timestamp, min_timestamp, max_timestamp)

    @staticmethod
    def _check_leave_n_out_range(df: pd.DataFrame, leave_n_out: int):
        """Validate that `leave_n_out` is within a reasonable range per user.

        The range is calculated as 10% to 90% of average interactions per user.

        Args:
            df (pd.DataFrame): DataFrame containing a 'userId' column.
            leave_n_out (int): Number of items to leave out for test.

        Raises:
            ValueError: If `leave_n_out` is outside the computed range.
        """
        user_count = df['userId'].nunique()
        interaction_count = len(df)

        min_leave_n_out = math.ceil(0.1 * interaction_count / user_count)
        max_leave_n_out = math.floor(0.9 * interaction_count / user_count)

        check_range('leave_n_out', leave_n_out, min_leave_n_out, max_leave_n_out)
