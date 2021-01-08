import unittest
import pandas as pd
from types import SimpleNamespace

from splitter.base_splitter import Splitter
"""
splitting:
    pre_split:
        train_path: ""
        validation_path: ""
        test_path: ""
    test_splitting:
        strategy: fixed_timestamp|temporal_hold_out|random_subsampling|random_cross_validation
        timestamp: best|1609786061
        test-ratio: 0.2
        leave-n-out: 1
        folds: 5
    validation_splitting:
        strategy: fixed_timestamp|temporal_hold_out|random_subsampling|random_cross_validation
        timestamp: best|1203300000
        test_ratio: 0.2
        leave_n_out: 1
        folds: 5
"""


class SplitterTest(unittest.TestCase):
    def setUp(self):
        # self.column_names = ['userId', 'itemId', 'rating', 'timestamp']
        self.column_names = ['userId', 'itemId', 'rating']
        self.data = pd.read_csv("../../data/categorical_dbpedia_ml1m/trainingset.tsv", sep="\t", header=None, names=self.column_names)

    # def test_read_files(self):
    #     files = SimpleNamespace(**{"train_path": "../../data/categorical_dbpedia_ml1m/trainingset.tsv",
    #                                "test_path": "../../data/categorical_dbpedia_ml1m/testset.tsv"
    #                                })
    #     self.split_ns =  SimpleNamespace(**{"pre_split": files})
    #     splitter = Splitter(self.data, self.split_ns)

    # def test_splitting(self):
    #     strategy = SimpleNamespace(**{"strategy": "fixed_timestamp",
    #                                "timestamp": "1203300000"
    #                                })
    #     self.split_ns = SimpleNamespace(**{"test_splitting": strategy})
    #     splitter = Splitter(self.data, self.split_ns)
    #     splitter.process_splitting()

    # def test_0(self):
    #     strategy_test = SimpleNamespace(**{"strategy": "temporal_hold_out",
    #                                        "test_ratio": 0.2
    #                                        })
    #     # strategy_val = SimpleNamespace(**{"strategy": "temporal_hold_out",
    #     #                            "test_ratio": 0.2
    #     #                            })
    #     self.split_ns = SimpleNamespace(**{"test_splitting": strategy_test})
    #     splitter = Splitter(self.data, self.split_ns)
    #     tuple_list = splitter.process_splitting()
    #     pass

    def test_1(self):
        strategy_test = SimpleNamespace(**{"strategy": "random_subsampling",
                                           "test_ratio": 0.2,
                                           "folds": 1
                                           })
        strategy_val = SimpleNamespace(**{"strategy": "random_subsampling",
                                          "test_ratio": 0.2,
                                           "folds": 1
                                          })
        self.split_ns = SimpleNamespace(**{"test_splitting": strategy_test, "validation_splitting": strategy_val})
        splitter = Splitter(self.data, self.split_ns)
        tuple_list = splitter.process_splitting()
        pass


    # def test_fill_model(self):
    #     self.name_sapce_model.fill_base()
    #     random_task = list(self.name_sapce_model.base_namespace.tasks.__dict__.keys())[0]
    #     res = list(self.name_sapce_model.fill_model(self.name_sapce_model.base_namespace.
    #                                                 tasks.__getattribute__(random_task)))
    #     self.assertEqual(-1, self.name_sapce_model.base_namespace.gpu)
    #     self.assertEqual('regressor_2020oct', res[0][0])
    #     self.assertEqual(1, len(res))