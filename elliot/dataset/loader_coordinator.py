from typing import List, Union, Dict, Tuple
from types import SimpleNamespace
import importlib
import numpy as np
import pandas as pd

from elliot.namespace import ExperimentConfig, DataConfig
from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.splitter.base_splitter import Splitter
from elliot.prefiltering.standard_prefilters import PreFilter
from elliot.dataset.dataset import DataSet
from elliot.utils.enums import DataLoadingStrategy, AlignmentMode
from elliot.utils.read import Reader
from elliot.utils import logging
from elliot.utils.registry import side_info_registry


class DataSetLoader:
    """The DataSetLoader class is responsible for loading and preparing datasets for training,
    validation, and testing.

    It supports multiple loading strategies and integrates optional pre-filtering and side information loading.
    The final output is a list of `DataSet` objects, ready to be consumed by the recommendation pipeline.

    Args:
        config (ExperimentConfig): Configuration namespace object defining data paths, splitting strategy,
            filters, etc.

    Supported Loading Strategies:

    - `fixed`: Load train/test/(optional) validation sets from files.
    - `hierarchy`: Load multiple folds from a nested directory structure.
    - `dataset`: Load a single dataset and later applies pre-filtering and splitting.

    To configure the data loading, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      data_config:
        strategy: dataset|fixed|hierarchy
        data_folder: this/is/the/path
        dataset_path: this/is/the/path
      binarize: True|False
        side_information:
          - dataloader: FeatureLoader1
            map: this/is/the/path.tsv
            features: this/is/the/path.tsv
            properties: this/is/the/path.conf
          - dataloader: FeatureLoader2
            folder_map_features: this/is/the/path/folder
    """

    data_config: DataConfig
    dataframe: Union[list, pd.DataFrame]
    side_information: Dict[str, AbstractLoader]

    def __init__(self, config: ExperimentConfig):
        self.logger = logging.get_logger(self.__class__.__name__)
        self.reader = Reader(self.logger)

        self.config = config
        self.data_config = config.data_config

        # Default to align side information with the observed training set when present
        if self.data_config.side_information:
            self.config.align_side_with_train = True

        if self.config.config_test:
            return

        self._load_ratings()
        self._load_side_information()
        self._preprocess_data()

    def _load_ratings(self):
        """Load user-item interaction data according to the selected strategy."""
        reader_config = self.data_config.reader

        match self.data_config.strategy:

            case DataLoadingStrategy.FIXED:
                self.dataframe = self.reader.read_tabular_split(
                    read_folder=self.data_config.data_folder,
                    header=reader_config.header,
                    columns=reader_config.column_names(),
                    datatypes=reader_config.column_dtypes(),
                    sep=reader_config.sep,
                    ext=reader_config.ext,
                    callback_fn=self._rename_cols_and_binarize
                )

            case DataLoadingStrategy.HIERARCHY:
                self.dataframe = self.reader.read_tabular_split(
                    read_folder=self.data_config.data_folder,
                    hierarchical=True,
                    header=reader_config.header,
                    columns=reader_config.column_names(),
                    datatypes=reader_config.column_dtypes(),
                    sep=reader_config.sep,
                    ext=reader_config.ext,
                    callback_fn=self._rename_cols_and_binarize
                )

            case DataLoadingStrategy.DATASET:
                self.dataframe = self.reader.read_tabular(
                    path=self.data_config.dataset_path,
                    header=reader_config.header,
                    columns=reader_config.column_names(),
                    datatypes=reader_config.column_dtypes(),
                    sep=reader_config.sep,
                    callback_fn=self._rename_cols_and_binarize
                )

        self._clean(self._filter_nan_and_duplicates)

    def _rename_cols_and_binarize(self, data, **kwargs):
        names = ["userId", "itemId", "rating", "timestamp"]
        current_names = self.data_config.reader.column_names()

        col_iter = iter(data.columns)
        current_names = [next(col_iter) if isinstance(c, int) else c for c in current_names]

        col_mapping = {c: names[i] for i, c in enumerate(current_names) if c in data.columns}

        cols_to_use = list(col_mapping.values())
        data.rename(columns=col_mapping, inplace=True)
        data = data[cols_to_use]

        if any(c not in data.columns for c in ("userId", "itemId")):
            raise KeyError("Missing some required columns: 'userId' or 'itemId'.")

        if self.config.binarize == True or "rating" not in data.columns:
            data["rating"] = 1.0

        return data

    def _load_side_information(self):
        """Load side information (e.g., user/item features) using custom dataloaders defined in config.

        Raises:
            TypeError: If a provided loader does not inherit from AbstractLoader.
        """
        users, items = set(), set()
        df = self.dataframe

        if isinstance(df, list):
            folds, train, test = df[0]
            users |= set(test["userId"].unique())
            items |= set(test["itemId"].unique())

            if train is None:
                tr, val = folds[0]

                users |= set(tr["userId"].unique())
                items |= set(tr["itemId"].unique())

                users |= set(val["userId"].unique())
                items |= set(val["itemId"].unique())
            else:
                users |= set(train["userId"].unique())
                items |= set(train["itemId"].unique())

        else:
            users = set(df["userId"].unique())
            items = set(df["itemId"].unique())

        self._users = users
        self._items = items

        side_info_objs = {}
        for side in self.data_config.side_information:
            side_obj = side_info_registry.get(
                name=side.dataloader,
                users=users,
                items=items,
                ns=side,
                logger=self.logger
            )
            side_info_objs[side_obj.name] = side_obj

        self.side_information = side_info_objs

    def _preprocess_data(self):
        """Apply user/item filtering based on side information, and basic cleanup.
        Perform optional pre-filtering.
        """
        self._intersect_users_items()
        self._clean(self._filter_users_and_items)
        self._maybe_materialize_cache()

        del self._items, self._users

        if self.data_config.strategy == DataLoadingStrategy.DATASET:
            prefilter = PreFilter(self.dataframe, self.config.prefiltering)
            self.dataframe = prefilter.filter()

    def _intersect_users_items(self):
        """Align users/items with side information based on alignment mode:
        - DROP: intersect with side info (current behavior)
        - PAD: keep full train set; side loaders can pad/UNK internally
        - IMPUTE: keep full train set; side loaders should impute defaults
        """
        users, items = self._users, self._items
        user_aligned = users.copy()
        item_aligned = items.copy()

        for side_obj in self.side_information.values():
            mode = side_obj.alignment
            s_users, s_items = side_obj.get_mapped()
            if mode == AlignmentMode.DROP:
                user_aligned &= s_users
                item_aligned &= s_items
            elif mode in (AlignmentMode.PAD, AlignmentMode.IMPUTE):
                # Keep full set; loaders handle padding/imputing internally
                pass

        # Apply filtering for DROP sources
        for side_obj in self.side_information.values():
            mode = side_obj.alignment
            if mode == AlignmentMode.DROP:
                side_obj.filter(user_aligned, item_aligned)

        self._users, self._items = user_aligned, item_aligned

    def _clean(self, clean_fn):
        """Clean all loaded DataFrames by filtering users/items and removing duplicates."""
        def clean(df): return clean_fn(df) if df is not None else None

        if isinstance(self.dataframe, list):
            new_dataframe = []
            for folds, tr, te in self.dataframe:
                test = clean(te)
                train = clean(tr)
                folds = [(clean(tr_), clean(va)) for tr_, va in folds]
                new_dataframe.append((folds, train, test))
            self.dataframe = new_dataframe
        else:
            self.dataframe = clean(self.dataframe)

    def _filter_nan_and_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a single DataFrame based on valid users/items and applies basic cleanup,
        i.e., handles missing values in the 'timestamp' column (if present), and removes duplicates.

        Args:
            df (pd.DataFrame): The DataFrame to clean.

        Returns:
            pd.DataFrame: Cleaned DataFrame.
        """
        mean_imputing_feats = ["timestamp"]
        for feat in mean_imputing_feats:
            if feat in list(df.columns):
                df[feat] = df[feat].fillna(df[feat].mean())

        df.dropna(inplace=True)
        df.drop_duplicates(keep='first', inplace=True)
        return df

    def _filter_users_and_items(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df[df["userId"].isin(self._users) & df["itemId"].isin(self._items)].reset_index(drop=True)
        return df

    def _maybe_materialize_cache(self):
        """Hook for large side-information sources: allow loaders to expose a
        preferred materialization strategy (lazy/memory/mmap). For now, we
        log intent; specific loaders can honor _materialization internally.
        """
        for side_obj in self.side_information.values():
            mat = side_obj.materialization
            if not mat:
                continue
            self.logger.debug(
                "Side-info materialization hint",
                extra={
                    "context": {
                        "source": side_obj.__class__.__name__,
                        "materialization": mat,
                        "alignment": side_obj.alignment,
                    }
                },
            )

    def build(self) -> Tuple[List[List[DataSet]], List[DataSet]]:
        if self.data_config.strategy != DataLoadingStrategy.DATASET:
            tuple_list = self.dataframe
        else:
            self.logger.info("There will be the splitting")
            splitter = Splitter(self.dataframe, self.config.splitting, self.config.random_seed)
            tuple_list = splitter.process_splitting()

        if len(tuple_list) > 1:
            self.logger.warning(
                "You are using a splitting strategy with folds. "
                "Paired TTest and Wilcoxon Test are not available!"
            )
            self.config.evaluation.paired_ttest = {}
            self.config.evaluation.wilcoxon_test = {}

        train_val_data, main_data = [], []

        for p1, (folds, original_train, test) in enumerate(tuple_list):
            # Test level
            self.logger.info(f"Test Fold {p1}")

            train = (
                original_train
                if original_train is not None else folds[0][0]
            )

            test_data_object = DataSet(
                config=self.config,
                train_data=train,
                eval_data=test,
                side_info_data=self.side_information,
                evaluation_set="test",
                fold_index=(p1, None)
            )
            main_data.append(test_data_object)

            val_list = []

            for p2, (train, val) in enumerate(folds):
                # Validation level
                self.logger.info(f"Test Fold {p1} - Validation Fold {p2}")

                train_data = (
                    test_data_object.train_set
                    if original_train is None else train
                )

                val_data_object = DataSet(
                    config=self.config,
                    train_data=train_data,
                    eval_data=val,
                    side_info_data=self.side_information,
                    evaluation_set="validation",
                    fold_index=(p1, p2)
                )

                val_list.append(val_data_object)

            if not val_list:
                val_list = [test_data_object]

            train_val_data.append(val_list)

        return train_val_data, main_data

    def prepare_dataset(self, val_data, main_data):
        self.logger.info("Preparing dataset for evaluation")

        for p1, (folds, main) in enumerate(zip(val_data, main_data)):
            # Test level
            self.logger.info(f"Test Fold {p1}")
            main.get_eval_dataloader()

            for p2, fold in enumerate(folds):
                # Validation level
                self.logger.info(f"Test Fold {p1} - Validation Fold {p2}")
                fold.get_eval_dataloader()


def build_mock_dataset(config) -> List[List[DataSet]]:
    names = ["userId", "itemId", "rating"]
    np.random.seed(config.random_seed)

    train_set = np.hstack((
        np.random.randint(0, 5 * 20, size=(5 * 20, 2)),
        np.random.randint(0, 2, size=(5 * 20, 1))
    ))
    test_set = np.hstack((
        np.random.randint(0, 5 * 20, size=(5 * 20, 2)),
        np.random.randint(0, 2, size=(5 * 20, 1))
    ))

    train_set = pd.DataFrame(np.array(train_set), columns=names)
    test_set = pd.DataFrame(np.array(test_set), columns=names)

    data_list = [[
        DataSet(
            config=config,
            data_tuple=(train_set, None, test_set),
            side_information_data=SimpleNamespace()
        )
    ]]

    return data_list
