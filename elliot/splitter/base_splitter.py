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

        if hasattr(splitting_ns, "save_on_disk"):
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
                exploded_train_list = []
                for single_train, single_test in tuple_list:
                    # [(train_0,test_0), (train_1,test_1), (train_2,test_2), (train_3,test_3), (train_4,test_4)]
                    train_val_test_tuples_list = self.handle_hierarchy(single_train,
                                                                       splitting_ns.validation_splitting)
                    exploded_train_list.append(train_val_test_tuples_list)
                tuple_list = self.rearrange_data(tuple_list, exploded_train_list)

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
        if hasattr(valtest_splitting_ns, "strategy"):
            if valtest_splitting_ns.strategy == "fixed_timestamp":
                if hasattr(valtest_splitting_ns, "timestamp"):
                    if valtest_splitting_ns.timestamp.isdigit():
                        tuple_list = self.splitting_passed_timestamp(data, int(valtest_splitting_ns.timestamp))
                    elif valtest_splitting_ns.timestamp == "best":
                        print("Here")
                        kwargs = {}
                        if hasattr(valtest_splitting_ns, "min_below"):
                            kwargs["min_below"] = int(valtest_splitting_ns.min_below)
                        if hasattr(valtest_splitting_ns, "min_over"):
                            kwargs["min_over"] = int(valtest_splitting_ns.min_over)
                        tuple_list = self.splitting_best_timestamp(data, **kwargs)

                    else:
                        raise Exception("Timestamp option value is not valid")
                else:
                    raise Exception(f"Option timestamp missing for {valtest_splitting_ns.strategy} strategy")
            elif valtest_splitting_ns.strategy == "temporal_hold_out":
                if hasattr(valtest_splitting_ns, "test_ratio"):
                    tuple_list = self.splitting_temporal_holdout(data, float(valtest_splitting_ns.test_ratio))
                elif hasattr(valtest_splitting_ns, "leave_n_out"):
                    tuple_list = self.splitting_temporal_leavenout(data, int(valtest_splitting_ns.leave_n_out))
                else:
                    raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")
            elif valtest_splitting_ns.strategy == "random_subsampling":
                if hasattr(valtest_splitting_ns, "folds"):
                    if str(valtest_splitting_ns.folds).isdigit():
                        pass
                    else:
                        raise Exception("Folds option value is not valid")
                else:
                    valtest_splitting_ns.folds = 1
                    print("Folds option value is missing. It has been set to 1")
                    # raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")

                if hasattr(valtest_splitting_ns, "test_ratio"):
                    tuple_list = self.splitting_randomsubsampling_kfolds(data, int(valtest_splitting_ns.folds),
                                                                         float(valtest_splitting_ns.test_ratio))
                elif hasattr(valtest_splitting_ns, "leave_n_out"):
                    tuple_list = self.splitting_randomsubsampling_kfolds_leavenout(data, int(valtest_splitting_ns.folds),
                                                                                   int(valtest_splitting_ns.leave_n_out))
                else:
                    raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")
            elif valtest_splitting_ns.strategy == "random_cross_validation":
                if hasattr(valtest_splitting_ns, "folds"):
                    if str(valtest_splitting_ns.folds).isdigit():
                        tuple_list = self.splitting_kfolds(data, int(valtest_splitting_ns.folds))
                    else:
                        raise Exception("Folds option value is not valid")
                else:
                    raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")
            else:
                raise Exception(f"Unrecognized Test Strategy:\t{valtest_splitting_ns.strategy}")
        else:
            raise Exception("Strategy option not found")

        return tuple_list  # it returns a list tuples (pairs) of train test dataframes

    def rearrange_data(self, train_test: t.List[t.Tuple[pd.DataFrame, pd.DataFrame]],
                       train_val: t.List[t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]]):
        return [(train_val[p], v[1]) for p, v in enumerate(train_test)]

    def generic_split_function(self, data: pd.DataFrame, **kwargs) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        pass

    def fold_list_generator(self, length, folds=5):
        def infinite_looper(folds=5):
            while True:
                for f in range(folds):
                    yield f

        looper = infinite_looper(folds)
        return [next(looper) for _ in range(length)]

    def splitting_kfolds(self, data: pd.DataFrame, folds=5):
        tuple_list = []
        user_groups = data.groupby(['userId'])
        for name, group in user_groups:
            data.loc[group.index, 'fold'] = self.fold_list_generator(len(group), folds)
        data["fold"] = pd.to_numeric(data["fold"], downcast='integer')
        for i in range(folds):
            test = data[data["fold"] == i].drop(columns=["fold"]).reset_index(drop=True)
            train = data[data["fold"] != i].drop(columns=["fold"]).reset_index(drop=True)
            tuple_list.append((train, test))
        return tuple_list

    def splitting_temporal_holdout(self, d: pd.DataFrame, ratio=0.2):
        tuple_list = []
        data = d.copy()
        user_size = data.groupby(['userId'], as_index=True).size()
        user_threshold = user_size.apply(lambda x: math.floor(x * (1 - ratio)))
        data['rank_first'] = data.groupby(['userId'])['timestamp'].rank(method='first', ascending=True, axis=1)
        data["test_flag"] = data.apply(
            lambda x: x["rank_first"] > user_threshold.loc[x["userId"]], axis=1)
        test = data[data["test_flag"] == True].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
        train = data[data["test_flag"] == False].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
        tuple_list.append((train, test))
        return tuple_list

    def splitting_temporal_leavenout(self, d: pd.DataFrame, n=1):
        tuple_list = []
        data = d.copy()
        data['rank_first'] = data.groupby(['userId'])['timestamp'].rank(method='first', ascending=False, axis=1)
        data["test_flag"] = data.apply(
            lambda x: x["rank_first"] <= n, axis=1)
        test = data[data["test_flag"] == True].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
        train = data[data["test_flag"] == False].drop(columns=["rank_first", "test_flag"]).reset_index(drop=True)
        tuple_list.append((train, test))
        return tuple_list

    def splitting_passed_timestamp(self, d: pd.DataFrame, timestamp=1):
        tuple_list = []
        data = d.copy()
        data["test_flag"] = data.apply(lambda x: x["timestamp"] >= timestamp, axis=1)
        test = data[data["test_flag"] == True].drop(columns=["test_flag"]).reset_index(drop=True)
        train = data[data["test_flag"] == False].drop(columns=["test_flag"]).reset_index(drop=True)
        tuple_list.append((train, test))
        return tuple_list

    def subsampling_list_generator(self, length, ratio=0.2):
        train = int(math.floor(length * (1 - ratio)))
        test = length - train
        list_ = [0] * train + [1] * test
        np.random.shuffle(list_)
        return list_

    def splitting_randomsubsampling_kfolds(self, d: pd.DataFrame, folds=5, ratio=0.2):
        tuple_list = []
        data = d.copy()
        user_groups = data.groupby(['userId'])
        for i in range(folds):
            for name, group in user_groups:
                data.loc[group.index, 'test_flag'] = self.subsampling_list_generator(len(group), ratio)
            data["test_flag"] = pd.to_numeric(data["test_flag"], downcast='integer')
            test = data[data["test_flag"] == 1].drop(columns=["test_flag"]).reset_index(drop=True)
            train = data[data["test_flag"] == 0].drop(columns=["test_flag"]).reset_index(drop=True)
            tuple_list.append((train, test))
        return tuple_list

    def subsampling_leavenout_list_generator(self, length, n=1):
        test = n
        train = length - test
        list_ = [0] * train + [1] * test
        np.random.shuffle(list_)
        return list_

    def splitting_randomsubsampling_kfolds_leavenout(self, d: pd.DataFrame, folds=5, n=1):
        tuple_list = []
        data = d.copy()
        user_groups = data.groupby(['userId'])
        for i in range(folds):
            for name, group in user_groups:
                data.loc[group.index, 'test_flag'] = self.subsampling_leavenout_list_generator(len(group), n)
            data["test_flag"] = pd.to_numeric(data["test_flag"], downcast='integer')
            test = data[data["test_flag"] == 1].drop(columns=["test_flag"]).reset_index(drop=True)
            train = data[data["test_flag"] == 0].drop(columns=["test_flag"]).reset_index(drop=True)
            tuple_list.append((train, test))
        return tuple_list

    def splitting_best_timestamp(self, d: pd.DataFrame, min_below=1, min_over=1):
        data = d.copy()
        unique_timestamps = data["timestamp"].unique()
        user_groups = data.groupby(['userId'])
        ts_dict = {}
        nuniques = len(unique_timestamps)
        i = 0
        for ts in unique_timestamps:
            print(nuniques - i)
            i += 1
            ts_dict[ts] = 0
            for name, group in user_groups:
                below = group[group["timestamp"] < ts]["timestamp"].count()
                over = len(group) - below
                if (below >= min_below) and (over >= min_over):
                    ts_dict[ts] += 1
        max_val = max(ts_dict.values())
        best_tie = [ts for ts,v in ts_dict.items() if v == max_val]
        max_ts = max(best_tie)
        print(f"Best Timestamp: {max_ts}")
        return self.splitting_passed_timestamp(d, max_ts)
