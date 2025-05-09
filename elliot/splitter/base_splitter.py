import typing as t
import pandas as pd
import numpy as np
import math
import shutil
import os

from types import SimpleNamespace

from elliot.utils.folder import create_folder_by_index

"""        
data_config:
    strategy: dataset|hierarchy|fixed
    dataset: example
    dataloader: KnowledgeChains
    dataset_path: "path"
    root_folder: "path"
    train_path: ""
    validation_path: ""
    test_path: ""
    side_information:
        feature_data: ../data/{0}/original/features.npy
        map: ../data/{0}/map.tsv
        features: ../data/{0}/features.tsv
        properties: ../data/{0}/properties.conf
    output_rec_result: ../results/{0}/recs/
    output_rec_weight: ../results/{0}/weights/
    output_rec_performance: ../results/{0}/performance/
splitting:
    save_on_disk: True
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
"""
"""
Nested Cross-Validation
[(train_0,test_0), (train_1,test_1), (train_2,test_2), (train_3,test_3), (train_4,test_4)]

[([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_0),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_1),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_2),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_3),
([(train_0,val_0), (train_1,val_1), (train_2,val_2), (train_3,val_3), (train_4,val_4)],test_4)]

Nested Hold-Out
[(train_0,test_0)]

[([(train_0,test_0)],test_0)]
"""


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
        save_folder (str or None): Directory path to store the splits if saving is enabled.

    Supported Splitting Strategies:
        - fixed_timestamp: Splits data using a predefined or computed timestamp threshold.
        - best_timestamp: Automatically selects the best timestamp across users for splitting.
        - temporal_hold_out: Uses the most recent interactions (per user) for testing.
        - temporal_leave_n_out: Leaves the last N events per user as the test set.
        - random_subsampling: Randomly subsamples training and testing data across folds.
        - random_cross_validation (K-Folds): Performs stratified K-Fold cross-validation.

    Raises:
        Exception: If required strategy parameters are missing or invalid.

    To configure the data splitting, include the appropriate
    settings in the configuration file using the pattern shown below.
    Note: Splitting is required and will be applied only if `data_config.strategy` is set to `"dataset"`.

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
    """

    def __init__(self, data: pd.DataFrame, splitting_ns: SimpleNamespace, random_seed=42):
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

    def process_splitting(self):
        """
        Executes the configured splitting strategy (Train/Test or Train/Validation/Test).

        Returns:
            list: A list of (train, test) or ((train, val), test) tuples.

        Raises:
            Exception: If required options such as test splitting or save paths are missing.
        """
        np.random.seed(self.random_seed)
        data = self.data
        splitting_ns = self.splitting_ns

        if getattr(splitting_ns, "save_on_disk", False):
            if hasattr(splitting_ns, "save_folder"):
                self.save_on_disk = True
                self.save_folder = splitting_ns.save_folder

                if os.path.exists(self.save_folder):
                    shutil.rmtree(self.save_folder, ignore_errors=True)

                os.makedirs(self.save_folder)
            else:
                raise Exception("Train or Test paths are missing")

        if hasattr(splitting_ns, "test_splitting"):
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
        else:
            raise Exception("Test splitting strategy is not defined")

        if self.save_on_disk:
            self.store_splitting(tuple_list)

        return tuple_list

    def store_splitting(self, tuple_list):
        """
        Saves the generated splits to disk as TSV files if enabled.

        Args:
            tuple_list (list): A list of split tuples to be saved on disk.
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

    # def read_folder(self, folder_path):
    #     for root, dirs, files in os.walk(folder_path):
    #         if not dirs:
    #             # leggi i due file
    #
    #             pass
    #         else:
    #             pass
    #         pass

    def handle_hierarchy(self, data: pd.DataFrame, valtest_splitting_ns: SimpleNamespace) -> t.List[
        t.Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Handles the splitting logic based on the selected strategy.

        Args:
            data (pd.DataFrame): The dataset to be split.
            valtest_splitting_ns (SimpleNamespace): Namespace containing strategy parameters.

        Returns:
            list: A list of (train, test) tuples.

        Raises:
            Exception: If the strategy or required parameters are not defined or invalid.
        """
        data = data.reset_index(drop=True)

        strategy = getattr(valtest_splitting_ns, "strategy", None)
        if not strategy:
            raise Exception("Strategy option not found")

        if strategy == "fixed_timestamp":
            timestamp = getattr(valtest_splitting_ns, "timestamp", None)
            if not timestamp:
                raise Exception(f"Option 'timestamp' missing for '{strategy}' strategy")

            if str(timestamp).isdigit():
                tuple_list = self.splitting_passed_timestamp(data, int(timestamp))
            elif timestamp == "best":
                kwargs = {}
                min_below = getattr(valtest_splitting_ns, "min_below", None)
                min_over = getattr(valtest_splitting_ns, "min_over", None)
                if min_below is not None:
                    kwargs["min_below"] = int(min_below)
                if min_over is not None:
                    kwargs["min_over"] = int(min_over)
                tuple_list = self.splitting_best_timestamp(data, **kwargs)
            else:
                raise Exception("Timestamp option value is not valid")

        elif strategy == "temporal_hold_out":
            test_ratio = getattr(valtest_splitting_ns, "test_ratio", None)
            leave_n_out = getattr(valtest_splitting_ns, "leave_n_out", None)

            if test_ratio is not None:
                tuple_list = self.splitting_temporal_holdout(data, float(test_ratio))
            elif leave_n_out is not None:
                tuple_list = self.splitting_temporal_leave_n_out(data, int(leave_n_out))
            else:
                raise Exception(f"Option missing for '{strategy}' strategy")

        elif strategy == "random_subsampling":
            folds = getattr(valtest_splitting_ns, "folds", 1)
            try:
                folds = int(folds)
            except ValueError:
                raise Exception("Folds option value is not valid")

            test_ratio = getattr(valtest_splitting_ns, "test_ratio", None)
            leave_n_out = getattr(valtest_splitting_ns, "leave_n_out", None)

            if test_ratio is not None:
                tuple_list = self.splitting_random_subsampling_k_folds(data, folds, float(test_ratio))
            elif leave_n_out is not None:
                tuple_list = self.splitting_random_subsampling_k_folds_leave_n_out(data, folds, int(leave_n_out))
            else:
                raise Exception(f"Option missing for '{strategy}' strategy")

        elif strategy == "random_cross_validation":
            folds = getattr(valtest_splitting_ns, "folds", None)
            if folds is None or not str(folds).isdigit():
                raise Exception("Folds option value is not valid")

            tuple_list = self.splitting_k_folds(data, int(folds))

        else:
            raise Exception(f"Unrecognized Test Strategy: {strategy}")

        return tuple_list

    # def generic_split_function(self, data: pd.DataFrame, **kwargs) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
    #     pass

    def splitting_temporal_holdout(self, d: pd.DataFrame, ratio=0.2):
        """
        Splits the dataset using temporal hold-out strategy.

        Args:
            d (pd.DataFrame): The dataset to split.
            ratio (float): Ratio of data to assign to the test set. Defaults to 0.2.

        Returns:
            list: A list containing one (train, test) tuple.
        """
        tuple_list = []
        user_size = d.groupby('userId').size()
        user_threshold = np.floor(user_size * (1 - ratio)).astype(int)

        rank = d.groupby('userId')['timestamp'].rank(method='first', ascending=True)
        mask = rank > d['userId'].map(user_threshold)

        tuple_list.append(self._split_with_mask(d, mask))

        return tuple_list

    def splitting_temporal_leave_n_out(self, d: pd.DataFrame, n=1):
        """
        Splits the dataset by leaving the last n items (per user) as test data.

        Args:
            d (pd.DataFrame): The dataset to split.
            n (int): Number of items per user to leave out for testing. Defaults to 1.

        Returns:
            list: A list containing one (train, test) tuple.
        """
        tuple_list = []

        rank = d.groupby('userId')['timestamp'].rank(method='first', ascending=False)
        mask = rank <= n

        tuple_list.append(self._split_with_mask(d, mask))

        return tuple_list

    def splitting_passed_timestamp(self, d: pd.DataFrame, timestamp=1):
        """
        Splits the dataset based on a fixed timestamp threshold.

        Args:
            d (pd.DataFrame): The dataset to split.
            timestamp (int): Timestamp threshold to separate train and test sets.

        Returns:
            list: A list containing one (train, test) tuple.
        """
        tuple_list = []

        mask = d['timestamp'] >= timestamp

        tuple_list.append(self._split_with_mask(d, mask))

        return tuple_list

    def splitting_best_timestamp(self, d: pd.DataFrame, min_below=1, min_over=1):
        """
        Splits the dataset based on the best timestamp, i.e., the one that maximizes user coverage.

        Args:
            d (pd.DataFrame): The dataset to split.
            min_below (int): Minimum number of interactions before the timestamp. Defaults to 1.
            min_over (int): Minimum number of interactions after the timestamp. Defaults to 1.

        Returns:
            list: A list containing one (train, test) tuple.

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
            raise ValueError("No valid timestamp found. Try lowering min_below or min_over.")

        max_votes = np.max(counter_array)
        max_indices = np.where(counter_array == max_votes)[0]
        best_ts = all_timestamps[max_indices[-1]]

        print(f"Best Timestamp: {best_ts}")
        return self.splitting_passed_timestamp(d, int(best_ts))

    def splitting_k_folds(self, d: pd.DataFrame, folds):
        """
        Performs K-Fold cross-validation splitting by user.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds.

        Returns:
            list: A list of (train, test) tuples, one per fold.
        """
        fold_indices = [[] for _ in range(folds)]
        user_groups = d.groupby('userId')

        def fold_list_generator(length):
            return [i % folds for i in range(length)]

        for _, group in user_groups:
            fold_ids = fold_list_generator(len(group))
            for idx, fold_id in zip(group.index, fold_ids):
                fold_indices[fold_id].append(idx)

        return self._split_k_folds(d, fold_indices=fold_indices)

    def splitting_random_subsampling_k_folds(self, d: pd.DataFrame, folds=5, ratio=0.2):
        """
        Performs random subsampling across multiple folds based on a test ratio.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds. Defaults to 5.
            ratio (float): Proportion of data per fold to use as test set. Defaults to 0.2.

        Returns:
            list: A list of (train, test) tuples.
        """
        def subsampling_list_generator(length):
            train = int(math.floor(length * (1 - ratio)))
            test = length - train
            list_ = [False] * train + [True] * test
            np.random.shuffle(list_)
            return list_

        return self._split_k_folds(d, subsampling_list_generator, folds=folds)

    def splitting_random_subsampling_k_folds_leave_n_out(self, d: pd.DataFrame, folds=5, n=1):
        """
        Performs random leave-n-out subsampling across multiple folds.

        Args:
            d (pd.DataFrame): The dataset to split.
            folds (int): Number of folds. Defaults to 5.
            n (int): Number of interactions per user to leave out for testing. Defaults to 1.

        Returns:
            list: A list of (train, test) tuples.
        """
        def subsampling_leave_n_out_list_generator(length):
            test = n
            train = length - test
            list_ = [False] * train + [True] * test
            np.random.shuffle(list_)
            return list_

        return self._split_k_folds(d, subsampling_leave_n_out_list_generator, folds=folds)

    def _split_k_folds(self, data, generator=None, folds=None, fold_indices=None):
        """
        Utility function to perform K-Fold splitting.

        If the method is called without the `fold_indices` parameter,
        both `folds` and `generator` must be provided.
        In this case, the splitting process will first compute in a standard way the fold indices
        and then apply the split.

        If the method is called with the `fold_indices` parameter
        (i.e., the split has already been precomputed in a custom way),
        the `folds` and `generator` parameters will be ignored,
        and the splitting will be applied directly using the provided indices.

        Args:
            data (pd.DataFrame): The dataset to split.
            generator (callable, optional): A function that generates a test mask.
            folds (int, optional): Number of folds.
            fold_indices (list, optional): Predefined indices for each fold.

        Returns:
            list: A list of (train, test) tuples.

        Raises:
            ValueError: If both folds and fold_indices are None, or generator is missing when required.
        """
        tuple_list = []

        if fold_indices is None:
            if folds is None:
                raise ValueError("Parameters 'folds' and 'fold_indices' cannot both be None")
            if generator is None:
                raise ValueError("Parameter 'generator' cannot be None if 'fold_indices' is None")
            fold_indices = [[] for _ in range(folds)]
            user_groups = data.groupby('userId')

            for fold_id in range(folds):
                for _, group in user_groups:
                    group_mask = generator(len(group))
                    fold_indices[fold_id].extend(group.index[group_mask])

        for i in range(len(fold_indices)):
            test_idx = fold_indices[i]
            mask = data.index.isin(test_idx)
            mask[test_idx] = True

            tuple_list.append(self._split_with_mask(data, mask))

        return tuple_list

    @staticmethod
    def _split_with_mask(data, mask):
        """
        Utility function to split the dataset into train and test using a boolean mask.

        Args:
            data (pd.DataFrame): The dataset to split.
            mask (pd.Series or np.array): Boolean array indicating the test set.

        Returns:
            tuple: A (train, test) tuple of DataFrames.
        """
        test = data[mask].reset_index(drop=True)
        train = data[~mask].reset_index(drop=True)
        return train, test
