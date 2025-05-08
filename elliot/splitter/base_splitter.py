import typing as t
import pandas as pd
import numpy as np
import math
import shutil
import os

from collections import Counter
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
    def __init__(self, data: pd.DataFrame, splitting_ns: SimpleNamespace, random_seed=42):
        self.random_seed = random_seed
        self.data = data
        self.splitting_ns = splitting_ns
        self.save_on_disk = False
        self.save_folder = None

    def process_splitting(self):
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
        tuple_list = []
        user_size = d.groupby('userId').size()
        user_threshold = np.floor(user_size * (1 - ratio)).astype(int)

        rank = d.groupby('userId')['timestamp'].rank(method='first', ascending=True)
        mask = rank > d['userId'].map(user_threshold)

        tuple_list.append(self._split_with_mask(d, mask))

        return tuple_list

    def splitting_temporal_leave_n_out(self, d: pd.DataFrame, n=1):
        tuple_list = []

        rank = d.groupby('userId')['timestamp'].rank(method='first', ascending=False)
        mask = rank <= n

        tuple_list.append(self._split_with_mask(d, mask))

        return tuple_list

    def splitting_passed_timestamp(self, d: pd.DataFrame, timestamp=1):
        tuple_list = []

        mask = d['timestamp'] >= timestamp

        tuple_list.append(self._split_with_mask(d, mask))

        return tuple_list

    def splitting_best_timestamp(self, d: pd.DataFrame, min_below=1, min_over=1):
        data = d.copy()
        data.sort_values(by=["userId", "timestamp"], inplace=True)

        ts_counter = Counter()
        user_groups = data.groupby("userId")

        for _, group in user_groups:
            timestamps = group["timestamp"].to_numpy()
            n = len(timestamps)
            # Skip users with not enough total events
            if n < (min_below + min_over):
                continue
            # Valid indices range
            start = min_below
            end = n - min_over
            if start >= end:
                continue  # No valid split point
            valid_timestamps = timestamps[start:end]
            ts_counter.update(valid_timestamps)

        if not ts_counter:
            raise ValueError("No valid timestamp found. Try lowering min_below or min_over.")

        max_votes = max(ts_counter.values())
        best_ts = max(ts for ts, count in ts_counter.items() if count == max_votes)

        print(f"Best Timestamp: {best_ts}")
        return self.splitting_passed_timestamp(d, best_ts)

    def splitting_k_folds(self, d: pd.DataFrame, folds):
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
        def subsampling_list_generator(length):
            train = int(math.floor(length * (1 - ratio)))
            test = length - train
            list_ = [False] * train + [True] * test
            np.random.shuffle(list_)
            return list_

        return self._split_k_folds(d, subsampling_list_generator, folds=folds)

    def splitting_random_subsampling_k_folds_leave_n_out(self, d: pd.DataFrame, folds=5, n=1):
        def subsampling_leave_n_out_list_generator(length):
            test = n
            train = length - test
            list_ = [False] * train + [True] * test
            np.random.shuffle(list_)
            return list_

        return self._split_k_folds(d, subsampling_leave_n_out_list_generator, folds=folds)

    def _split_k_folds(self, data, generator=None, folds=None, fold_indices=None):
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
        test = data[mask].reset_index(drop=True)
        train = data[~mask].reset_index(drop=True)
        return train, test
