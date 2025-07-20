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
    The DataSetLoader class is responsible for loading and preparing datasets for training, validation, and testing.

    It supports multiple loading strategies (`"fixed"`, `"hierarchy"`, `"dataset"`) and integrates optional
    pre-filtering and side information loading. The final output is a list of `DataSet` objects, ready to be
    consumed by the recommendation pipeline.

    Attributes:
        config (SimpleNamespace): Configuration namespace object defining data paths, splitting strategy, filters, etc.
        args (tuple): Additional positional arguments.
        kwargs (dict): Additional keyword arguments.
        column_names (list): Default column names used for reading interaction files.
        logger (Logger): Logger instance for the class.
        tuple_list (list): Contains train-validation-test splits depending on the strategy.
        dataframe (pd.DataFrame): DataFrame with interactions (only for `"dataset"` strategy).
        side_information (SimpleNamespace): Loaded side information, if specified.

    Supported Loading Strategies:
        - fixed: Loads train/test/(optional) validation sets from files.
        - hierarchy: Loads multiple folds from a nested directory structure.
        - dataset: Loads a single dataset and later applies pre-filtering and splitting.

    To configure the data loading, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      data_config:
        strategy: dataset|fixed|hierarchy
        dataset_path: this/is/the/path.tsv
        root_folder: this/is/the/path
        train_path: this/is/the/path.tsv
        validation_path: this/is/the/path.tsv
        test_path: this/is/the/path.tsv
      binarize: True|False
        side_information:
          - dataloader: FeatureLoader1
            map: this/is/the/path.tsv
            features: this/is/the/path.tsv
            properties: this/is/the/path.conf
          - dataloader: FeatureLoader2
            folder_map_features: this/is/the/path/folder
    """

    def __init__(self, config, *args, **kwargs):
        """
        Initializes the DataSetLoader object.

        Args:
            config (SimpleNamespace): Configuration namespace object.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """
        self.logger = logging.get_logger(self.__class__.__name__)
        self.args = args
        self.kwargs = kwargs
        self.config = config
        self.column_names = ['userId', 'itemId', 'rating', 'timestamp']
        self.dataframe = None
        self.tuple_list = None
        self.side_information = None

        self._load_data()

    def _load_data(self):
        """
        Fully loads and preprocesses the dataset.

        Executes loading of ratings, optional side information, and dataset preprocessing.
        """
        if self.config.config_test:
            return

        self._load_ratings()
        self._load_side_information()
        self._preprocess_data()

        if isinstance(self.tuple_list[0][1], list):
            self.logger.warning("You are using a splitting strategy with folds. "
                                "Paired TTest and Wilcoxon Test are not available!")
            self.config.evaluation.paired_ttest = False
            self.config.evaluation.wilcoxon_test = False

    def _load_ratings(self):
        """
        Loads user-item interaction data according to the selected strategy.

        Raises:
            Exception: If an unsupported strategy is specified.
        """
        if self.config.data_config.strategy == "fixed":
            path_train_data = self.config.data_config.train_path
            path_val_data = getattr(self.config.data_config, "validation_path", None)
            path_test_data = self.config.data_config.test_path

            train_df = self._read_from_tsv(path_train_data, check_timestamp=True)
            test_df = self._read_from_tsv(path_test_data, check_timestamp=True)

            self.logger.info(f"{path_train_data} - Loaded")

            if path_val_data:
                val_df = self._read_from_tsv(path_val_data, check_timestamp=True)
                self.dataframe = [([(train_df, val_df)], test_df)]
            else:
                self.dataframe = [(train_df, test_df)]

        elif self.config.data_config.strategy == "hierarchy":
            self.dataframe = self._read_splitting(self.config.data_config.root_folder)

        elif self.config.data_config.strategy == "dataset":
            path_dataset = self.config.data_config.dataset_path

            self.dataframe = self._read_from_tsv(path_dataset, check_timestamp=True)
            # self.logger.info(('{0} - Loaded'.format(path_dataset)))

        else:
            raise Exception("Strategy option not recognized")

    def _read_from_tsv(self, file_path, check_timestamp=False):
        """
        Loads a TSV file and optionally processes the timestamp or binarizes ratings.

        Args:
            file_path (str): Path to the TSV file containing interactions.
            check_timestamp (bool): Whether to drop the timestamp column if it's empty.

        Returns:
            pd.DataFrame: The loaded and formatted DataFrame.
        """
        df = pd.read_csv(file_path, sep='\t', header=None, names=self.column_names)
        if check_timestamp and all(df["timestamp"].isna()):
            df.drop(columns=["timestamp"], inplace=True)
        if self.config.binarize == True or all(df["rating"].isna()):
            df["rating"] = 1
        return df

    def _read_splitting(self, folder_path):
        """
        Reads train/val/test splits organized in a hierarchical folder structure.

        Args:
            folder_path (str): Root folder path containing the splits.

        Returns:
            list: A nested list of (train, val, test) splits for each fold.
        """
        tuple_list = []
        for dirs in os.listdir(folder_path):
            for test_dir in dirs:
                test_ = self._read_from_tsv(os.sep.join([folder_path, test_dir, "test.tsv"]))
                val_dirs = [os.sep.join([folder_path, test_dir, val_dir]) for val_dir in
                            os.listdir(os.sep.join([folder_path, test_dir])) if
                            os.path.isdir(os.sep.join([folder_path, test_dir, val_dir]))]
                val_list = []
                for val_dir in val_dirs:
                    train_ = self._read_from_tsv(os.sep.join([val_dir, "train.tsv"]))
                    val_ = self._read_from_tsv(os.sep.join([val_dir, "val.tsv"]))
                    val_list.append((train_, val_))
                if not val_list:
                    val_list = self._read_from_tsv(os.sep.join([folder_path, test_dir, "train.tsv"]))
                tuple_list.append((val_list, test_))

        return tuple_list

    def _load_side_information(self):
        """
        Loads side information (e.g., user/item features) using custom dataloaders defined in config.

        Raises:
            TypeError: If a provided loader does not inherit from AbstractLoader.
        """
        users, items = set(), set()
        df = self.dataframe

        if isinstance(df, list):
            train, test = df[0]
            users |= set(test["userId"].unique())
            items |= set(test["itemId"].unique())
            if isinstance(train, list):
                tr, val = train[0]
                users |= set(tr["userId"].unique()) | set(val["userId"].unique())
                items |= set(tr["itemId"].unique()) | set(val["itemId"].unique())
            else:
                users |= set(train["userId"].unique())
                items |= set(train["itemId"].unique())
        else:
            users = set(df["userId"].unique())
            items = set(df["itemId"].unique())

        self._users = users
        self._items = items

        side_info_objs = []
        sides = self.config.data_config.side_information
        for side in sides:
            module = importlib.import_module("elliot.dataset.dataloader.loaders")
            dataloader_class = getattr(module, side.dataloader)
            if not issubclass(dataloader_class, AbstractLoader):
                raise Exception("Custom Loaders must inherit from AbstractLoader")
            side_obj = dataloader_class(users, items, side, self.logger)
            side_info_objs.append(side_obj)

        self._side_info_objs = side_info_objs
        self._build_side_info_namespace()

    def _build_side_info_namespace(self):
        """
        Builds a unified namespace from all loaded side information objects.
        """
        ns = SimpleNamespace()
        for side_obj in self._side_info_objs:
            side_ns = side_obj.create_namespace()
            name = side_ns.__name__
            setattr(ns, name, side_ns)
        self.side_information = ns

    def _preprocess_data(self):
        """
        Applies user/item filtering based on side information, and basic cleanup.
        Performs optional pre-filtering, and dataset splitting, only if the `"dataset"` strategy is used.
        """
        self._intersect_users_items()
        self._clean()

        del self._items, self._users, self._side_info_objs

        if self.config.data_config.strategy != 'dataset':
            self.tuple_list = self.dataframe
            return

        if hasattr(self.config, 'prefiltering'):
            prefilter = PreFilter(self.dataframe, self.config.prefiltering)
            self.dataframe = prefilter.filter()

        self.logger.info("There will be the splitting")
        splitter = Splitter(self.dataframe, self.config.splitting, self.config.random_seed)
        self.tuple_list = splitter.process_splitting()

    def _intersect_users_items(self):
        """
        Repeatedly intersects users/items with those available in side information.
        """
        users, items = self._users, self._items
        users_items = [side_obj.get_mapped() for side_obj in self._side_info_objs]

        while True:
            new_users, new_items = users.copy(), items.copy()
            for us_, is_ in users_items:
                new_users &= us_
                new_items &= is_
            if len(new_users) == len(users) and len(new_items) == len(items):
                break
            users = new_users
            items = new_items
            for side_obj in self._side_info_objs:
                side_obj.filter(users, items)

        self._users, self._items = users, items

    def _clean(self):
        """
        Cleans all loaded DataFrames by filtering users/items and removing duplicates.

        Returns:
            Union[list, pd.DataFrame]: Cleaned dataset(s).
        """
        def clean(df): return self._clean_single_dataframe(df)

        if isinstance(self.dataframe, list):
            new_dataframe = []
            for tr, te in self.dataframe:
                test = clean(te)
                if isinstance(tr, list):
                    train_fold = [(clean(tr_), clean(va)) for tr_, va in tr]
                else:
                    train_fold = clean(tr)
                new_dataframe.append((train_fold, test))
            self.dataframe = new_dataframe
        else:
            self.dataframe = clean(self.dataframe)

    def _clean_single_dataframe(self, df):
        """
        Filters a single DataFrame based on valid users/items and applies basic cleanup,
        i.e., handles missing values in the 'timestamp' column (if present), and removes duplicates.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        df = df[df["userId"].isin(self._users) & df["itemId"].isin(self._items)].reset_index(drop=True)

        mean_imputing_feats = ['timestamp']
        for feat in mean_imputing_feats:
            if feat in list(df.columns):
                df[feat] = df[feat].fillna(df[feat].mean())

        df.dropna(inplace=True)
        df.drop_duplicates(keep='first', inplace=True)
        return df

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
