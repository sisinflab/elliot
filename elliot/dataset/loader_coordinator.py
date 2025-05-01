import os
import importlib
import typing as t
import numpy as np
import pandas as pd
from types import SimpleNamespace

from elliot.dataset.dataloader.abstract_loader import AbstractLoader
from elliot.utils import logging
from elliot.splitter.base_splitter import Splitter
from elliot.prefiltering.standard_prefilters import PreFilter
from elliot.dataset.dataset import DataSet


class DataSetLoader:
    """
    Dataset loader.

    Load ratings according to the provided strategy.
    (to be developed) Load item features, if required
    """

    def __init__(self, config, *args, **kwargs):
        self.logger = logging.get_logger(self.__class__.__name__)
        self.args = args
        self.kwargs = kwargs
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating', 'timestamp']
        if config.config_test:
            return

        self._load_ratings()
        self._load_additional_features()
        self._preprocess_data()

        if isinstance(self.tuple_list[0][1], list):
            self.logger.warning("You are using a splitting strategy with folds. "
                                "Paired TTest and Wilcoxon Test are not available!")
            self.config.evaluation.paired_ttest = False
            self.config.evaluation.wilcoxon_test = False

    def _load_ratings(self):
        """
        User interactions loader.

        Strategy:
        - fixed: read train, test and val (optional) data
        - hierarchy: read splitting
        - dataset: read dataset and perform pre-filtering and splitting
        """
        if self.config.data_config.strategy == "fixed":
            path_train_data = self.config.data_config.train_path
            path_val_data = getattr(self.config.data_config, "validation_path", None)
            path_test_data = self.config.data_config.test_path

            self.train_dataframe = self._load_data(path_train_data, check_timestamp=True)
            self.test_dataframe = self._load_data(path_test_data, check_timestamp=True)

            self.logger.info(f"{path_train_data} - Loaded")

            if path_val_data:
                self.validation_dataframe = self._load_data(path_val_data, check_timestamp=True)
                self.tuple_list = [([(self.train_dataframe, self.validation_dataframe)], self.test_dataframe)]
            else:
                self.tuple_list = [(self.train_dataframe, self.test_dataframe)]

        elif self.config.data_config.strategy == "hierarchy":
            self.tuple_list = self.read_splitting(self.config.data_config.root_folder)

        elif self.config.data_config.strategy == "dataset":
            path_dataset = self.config.data_config.dataset_path

            self.dataframe = self._load_data(path_dataset, check_timestamp=True)
            # self.logger.info(('{0} - Loaded'.format(path_dataset)))

        else:
            raise Exception("Strategy option not recognized")

    def _load_additional_features(self):
        dataframe = self.dataframe if hasattr(self, 'dataframe') else self.tuple_list
        self.dataframe, self.side_information = self.coordinate_information(dataframe,
                                                                            sides=self.config.data_config.side_information,
                                                                            logger=self.logger)
        # pass

    def _preprocess_data(self):
        if self.config.data_config.strategy != "dataset":
            return
        self.dataframe = PreFilter.filter(self.dataframe, self.config)
        self.logger.info("There will be the splitting")
        splitter = Splitter(self.dataframe, self.config.splitting, self.config.random_seed)
        self.tuple_list = splitter.process_splitting()

    def _load_data(self, file_path, check_timestamp=False):
        dataframe = pd.read_csv(file_path, sep='\t', header=None, names=self.column_names)
        if check_timestamp and all(dataframe["timestamp"].isna()):
            dataframe = dataframe.drop(columns=["timestamp"]).reset_index(drop=True)
        if self.config.binarize == True or all(dataframe["rating"].isna()):
            dataframe["rating"] = 1
        return dataframe

    def read_splitting(self, folder_path):
        tuple_list = []
        for dirs in os.listdir(folder_path):
            for test_dir in dirs:
                test_ = self._load_data(os.sep.join([folder_path, test_dir, "test.tsv"]))
                val_dirs = [os.sep.join([folder_path, test_dir, val_dir]) for val_dir in
                            os.listdir(os.sep.join([folder_path, test_dir])) if
                            os.path.isdir(os.sep.join([folder_path, test_dir, val_dir]))]
                val_list = []
                for val_dir in val_dirs:
                    train_ = self._load_data(os.sep.join([val_dir, "train.tsv"]))
                    val_ = self._load_data(os.sep.join([val_dir, "val.tsv"]))
                    val_list.append((train_, val_))
                if not val_list:
                    val_list = self._load_data(os.sep.join([folder_path, test_dir, "train.tsv"]))
                tuple_list.append((val_list, test_))

        return tuple_list

    def generate_dataobjects(self) -> t.List[object]:
        data_list = []
        for p1, (train_val, test) in enumerate(self.tuple_list):
            # testset level
            if isinstance(train_val, list):
                # validation level
                val_list = []
                for p2, (train, val) in enumerate(train_val):
                    self.logger.info(f"Test Fold {p1} - Validation Fold {p2}")
                    single_dataobject = DataSet(self.config, (train, val, test), self.side_information, self.args,
                                                self.kwargs)
                    val_list.append(single_dataobject)
                data_list.append(val_list)
            else:
                self.logger.info(f"Test Fold {p1}")
                single_dataobject = DataSet(self.config, (train_val, test), self.side_information, self.args,
                                            self.kwargs)
                data_list.append([single_dataobject])
        return data_list

    def generate_dataobjects_mock(self) -> t.List[object]:
        np.random.seed(self.config.random_seed)
        _column_names = ['userId', 'itemId', 'rating']
        training_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))
        test_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))

        training_set = pd.DataFrame(np.array(training_set), columns=_column_names)
        test_set = pd.DataFrame(np.array(test_set), columns=_column_names)
        data_list = [[DataSet(self.config, (training_set, test_set), self.args, self.kwargs)]]

        return data_list

    def coordinate_information(self, dataframe: t.Union[pd.DataFrame, t.List],
                               sides: t.List[SimpleNamespace]=[],
                               logger: object = None) -> t.Tuple[pd.DataFrame, SimpleNamespace]:
        if isinstance(dataframe, list):
            users = set()
            items = set()
            train, test = dataframe[0]
            users = users | set(test["userId"].unique())
            items = items | set(test["itemId"].unique())
            if not isinstance(train, list):
                users = users | set(train["userId"].unique())
                items = items | set(train["itemId"].unique())
            else:
                train, val = train[0]
                users = users | set(train["userId"].unique())
                items = items | set(train["itemId"].unique())
                users = users | set(val["userId"].unique())
                items = items | set(val["itemId"].unique())
        else:
            users = set(dataframe["userId"].unique())
            items = set(dataframe["itemId"].unique())

        ns = SimpleNamespace()

        side_info_objs = []
        users_items = []
        for side in sides:
            dataloader_class = getattr(importlib.import_module("elliot.dataset.dataloader.loaders"), side.dataloader)
            if issubclass(dataloader_class, AbstractLoader):
                side_obj = dataloader_class(users, items, side, logger)
                side_info_objs.append(side_obj)
                users_items.append(side_obj.get_mapped())
            else:
                raise Exception("Custom Loaders must inherit from AbstractLoader")

        while True:
            new_users = users
            new_items = items
            for us_, is_ in users_items:
                new_users = new_users & us_
                new_items = new_items & is_
            if (len(new_users) == len(users)) & (len(new_items) == len(items)):
                break
            else:
                users = new_users
                items = new_items

                for side_obj in side_info_objs:
                    side_obj.filter(users, items)

        for side_obj in side_info_objs:
            side_ns = side_obj.create_namespace()
            name = side_ns.__name__
            setattr(ns, name, side_ns)

        if isinstance(dataframe, list):
            new_dataframe = []
            for tr, te in dataframe:
                test = self.clean_dataframe(te, users, items)
                if isinstance(tr, list):
                    train_fold = []
                    for tr_, va in tr:
                        tr_ = self.clean_dataframe(tr_, users, items)
                        va = self.clean_dataframe(va, users, items)
                        train_fold.append((tr_, va))
                else:
                    train_fold = self.clean_dataframe(tr, users, items)
                new_dataframe.append((train_fold, test))
            dataframe = new_dataframe
            # dataframe = [([(self.clean_dataframe(tr_, users, items), self.clean_dataframe(va, users, items)) for tr_, va in tr], self.clean_dataframe(te, users, items)) for tr, te in dataframe]
        else:
            dataframe = self.clean_dataframe(dataframe, users, items)

        return dataframe, ns

    def clean_dataframe(self, dataframe, users, items):
        dataframe = dataframe[dataframe['userId'].isin(users)]
        return dataframe[dataframe['itemId'].isin(items)]