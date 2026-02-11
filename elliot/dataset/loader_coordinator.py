from typing import List, Union
from types import SimpleNamespace
import importlib
import numpy as np
import pandas as pd

from elliot.namespace import ExperimentConfig, DataConfig
from elliot.dataset.dataloader.abstract_loader import AbstractLoader
from elliot.dataset.dataloader.side_info_registry import (
    AlignmentMode,
    Materialization,
    side_info_registry,
)
from elliot.utils import logging
from elliot.splitter.base_splitter import Splitter
from elliot.prefiltering.standard_prefilters import PreFilter
from elliot.dataset.dataset import DataSet
from elliot.utils.enums import DataLoadingStrategy
from elliot.utils.read import Reader

reader = Reader()


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
    interactions_df: Union[list, pd.DataFrame]
    tuple_list: list
    side_information: SimpleNamespace

    def __init__(self, config: ExperimentConfig):
        self.logger = logging.get_logger(self.__class__.__name__)
        reader.logger = self.logger

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
                self.interactions_df = reader.read_tabular_split(
                    read_folder=self.data_config.data_folder,
                    header=reader_config.header,
                    columns=reader_config.column_names(),
                    datatypes=reader_config.column_dtypes(),
                    sep=reader_config.sep,
                    ext=reader_config.ext,
                    callback_fn=self._rename_cols_and_binarize
                )

            case DataLoadingStrategy.HIERARCHY:
                self.interactions_df = reader.read_tabular_split(
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
                self.interactions_df = reader.read_tabular(
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
        df = self.interactions_df

        if isinstance(df, list):
            train_val, test = df[0]
            users |= set(test["userId"].unique())
            items |= set(test["itemId"].unique())

            train, val = train_val[0]
            users |= set(train["userId"].unique())
            items |= set(train["itemId"].unique())
            if val is not None:
                users |= set(val["userId"].unique())
                items |= set(val["itemId"].unique())
        else:
            users = set(df["userId"].unique())
            items = set(df["itemId"].unique())

        self._users = users
        self._items = items

        side_info_objs = []
        sides = self.data_config.side_information
        for side in sides:
            module = importlib.import_module("elliot.dataset.dataloader.loaders")
            dataloader_class = getattr(module, side.dataloader)
            if not issubclass(dataloader_class, AbstractLoader):
                raise TypeError("Custom Loaders must inherit from AbstractLoader")
            desc = side_info_registry.get(side.dataloader)
            side_obj = dataloader_class(users, items, side, self.logger)
            materialization = getattr(side, "materialization", None) or (desc.materialization if desc else None)
            alignment = getattr(side, "alignment", None) or (desc.alignment if desc else AlignmentMode.DROP)
            setattr(side_obj, "_alignment_mode", alignment)
            setattr(side_obj, "_materialization", materialization)
            side_info_objs.append(side_obj)

        self._side_info_objs = side_info_objs
        self._build_side_info_namespace()

    def _build_side_info_namespace(self):
        """Build a unified namespace from all loaded side information objects."""
        ns = SimpleNamespace()
        for side_obj in self._side_info_objs:
            side_ns = side_obj.create_namespace()
            name = side_ns.__name__
            setattr(ns, name, side_ns)
        self.side_information = ns

    def _preprocess_data(self):
        """Apply user/item filtering based on side information, and basic cleanup.
        Perform optional pre-filtering.
        """
        self._intersect_users_items()
        self._clean(self._filter_users_and_items)
        self._maybe_materialize_cache()

        del self._items, self._users, self._side_info_objs

        prefilter = PreFilter(self.interactions_df, self.config.prefiltering)
        self.interactions_df = prefilter.filter()

    def _intersect_users_items(self):
        """Align users/items with side information based on alignment mode:
        - DROP: intersect with side info (current behavior)
        - PAD: keep full train set; side loaders can pad/UNK internally
        - IMPUTE: keep full train set; side loaders should impute defaults
        """
        users, items = self._users, self._items
        user_aligned = users.copy()
        item_aligned = items.copy()

        for side_obj in self._side_info_objs:
            mode = getattr(side_obj, "_alignment_mode", AlignmentMode.DROP)
            s_users, s_items = side_obj.get_mapped()
            if mode == AlignmentMode.DROP:
                user_aligned &= s_users
                item_aligned &= s_items
            elif mode in (AlignmentMode.PAD, AlignmentMode.IMPUTE):
                # Keep full set; loaders handle padding/imputing internally
                pass
            else:
                user_aligned &= s_users
                item_aligned &= s_items

        # Apply filtering for DROP sources
        for side_obj in self._side_info_objs:
            mode = getattr(side_obj, "_alignment_mode", AlignmentMode.DROP)
            if mode == AlignmentMode.DROP:
                side_obj.filter(user_aligned, item_aligned)

        self._users, self._items = user_aligned, item_aligned

    def _clean(self, clean_fn):
        """Clean all loaded DataFrames by filtering users/items and removing duplicates."""
        def clean(df): return clean_fn(df) if df is not None else None

        if isinstance(self.interactions_df, list):
            new_dataframe = []
            for tr, te in self.interactions_df:
                test = clean(te)
                train_fold = [(clean(tr_), clean(va)) for tr_, va in tr]
                new_dataframe.append((train_fold, test))
            self.interactions_df = new_dataframe
        else:
            self.interactions_df = clean(self.interactions_df)

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
        for side_obj in self._side_info_objs:
            mat = getattr(side_obj, "_materialization", None)
            if not mat:
                continue
            self.logger.debug(
                "Side-info materialization hint",
                extra={
                    "context": {
                        "source": side_obj.__class__.__name__,
                        "materialization": mat,
                        "alignment": getattr(side_obj, "_alignment_mode", None),
                    }
                },
            )

    def build(self) -> List[List[DataSet]]:
        if self.data_config.strategy != DataLoadingStrategy.DATASET:
            tuple_list = self.interactions_df
        else:
            self.logger.info("There will be the splitting")
            splitter = Splitter(self.interactions_df, self.config.splitting, self.config.random_seed)
            tuple_list = splitter.process_splitting()

        if len(tuple_list) > 1:
            self.logger.warning("You are using a splitting strategy with folds. "
                                "Paired TTest and Wilcoxon Test are not available!")
            self.config.evaluation.paired_ttest = {}
            self.config.evaluation.wilcoxon_test = {}

        data_list = []

        for p1, (train_val, test) in enumerate(tuple_list):
            # Test level
            val_list = []
            for p2, (train, val) in enumerate(train_val):
                # Validation level
                self.logger.info(
                    f"Test Fold {p1}{f" - Validation Fold {p2}" if val is not None else ""}"
                )
                single_data_object = DataSet(
                    config=self.config,
                    data_tuple=(train, val, test),
                    side_information_data=self.side_information
                )
                val_list.append(single_data_object)
            data_list.append(val_list)

        return data_list


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
