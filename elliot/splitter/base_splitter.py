import typing as t
import pandas as pd
from types import SimpleNamespace


"""
splitting:
    pre_split:
        train_path: ""
        validation_path: ""
        test_path: ""
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
    def __init__(self, data: pd.DataFrame, splitting_ns: SimpleNamespace):
        if hasattr(splitting_ns, "pre_split"):
            if hasattr(splitting_ns.pre_split, "train_path") and hasattr(splitting_ns.pre_split, "test_path"):
                if hasattr(splitting_ns.pre_split, "validation_path"):
                    print("Train\tValidation\tTest")
                else:
                    print("Train\tTest")
            else:
                raise Exception("Train or Test paths are missing")
        else:
            if hasattr(splitting_ns, "test_splitting"):
                # [(train_0,test_0), (train_1,test_1), (train_2,test_2), (train_3,test_3), (train_4,test_4)]
                train_test_tuples_list = self.handle_hierarchy(data, splitting_ns.test_splitting)

                if hasattr(splitting_ns, "validation_splitting"):
                    # TODO: check matching attributes
                    exploded_train_list = []
                    for single_train, single_test in train_test_tuples_list:
                        # [(train_0,test_0), (train_1,test_1), (train_2,test_2), (train_3,test_3), (train_4,test_4)]
                        train_val_test_tuples_list = self.handle_hierarchy(single_train, splitting_ns.test_splitting)
                        exploded_train_list.append(train_val_test_tuples_list)
                    exploded_data = self.rearrange_data(train_test_tuples_list, exploded_train_list)

                    # if hasattr(splitting_ns.validation_splitting, "strategy"):
                    #     pass
                    # else:
                    #     raise Exception("Validation Strategy not found")

                    print("Train\tValidation\tTest\tstrategies")
                else:
                    print("Train\tTest\tstrategies")
            else:
                raise Exception("Test splitting strategy is not defined")

    def handle_hierarchy(self, data: pd.DataFrame, valtest_splitting_ns: SimpleNamespace) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        if hasattr(valtest_splitting_ns, "strategy"):
            if valtest_splitting_ns.strategy == "fixed_timestamp":
                if hasattr(valtest_splitting_ns, "timestamp"):
                    if valtest_splitting_ns.timestamp.isdigit():
                        pass
                    elif valtest_splitting_ns.timestamp == "best":
                        print("Here")
                        pass
                    else:
                        raise Exception("Timestamp option value is not valid")
                else:
                    raise Exception(f"Option timestamp missing for {valtest_splitting_ns.strategy} strategy")
            elif valtest_splitting_ns.strategy == "temporal_hold_out":
                if hasattr(valtest_splitting_ns, "test_ratio"):
                    pass
                elif hasattr(valtest_splitting_ns, "leave_n_out"):
                    pass
                else:
                    raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")
            elif valtest_splitting_ns.strategy == "random_subsampling":
                if hasattr(valtest_splitting_ns, "folds"):
                    if valtest_splitting_ns.timestamp.isdigit():
                        pass
                    else:
                        raise Exception("Folds option value is not valid")
                else:
                    raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")

                if hasattr(valtest_splitting_ns, "test_ratio"):
                    pass
                elif hasattr(valtest_splitting_ns, "leave_n_out"):
                    pass
                else:
                    raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")
            elif valtest_splitting_ns.strategy == "random_cross_validation":
                if hasattr(valtest_splitting_ns, "test_ratio"):
                    pass
                elif hasattr(valtest_splitting_ns, "leave_n_out"):
                    pass
                else:
                    raise Exception(f"Option missing for {valtest_splitting_ns.strategy} strategy")
            else:
                raise Exception(f"Unrecognized Test Strategy:\t{valtest_splitting_ns.strategy}")
        else:
            raise Exception("Test Strategy not found")

        return [] # it returns a list tuples (pairs) of train test dataframes

    def rearrange_data(self, train_test: t.List[t.Tuple[pd.DataFrame, pd.DataFrame]], train_val:t.List[t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]]):
        return [(train_val[p], v) for p, v in enumerate(train_test)]

    def generic_split_function(self, data: pd.DataFrame, **kwargs) -> t.List[t.Tuple[pd.DataFrame, pd.DataFrame]]:
        pass