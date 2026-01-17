from typing import List, Union
from types import SimpleNamespace
import importlib
import numpy as np
import pandas as pd

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
from elliot.utils.config import DataLoadingConfig

reader = Reader()


class DataSetLoader:
    """The DataSetLoader class is responsible for loading and preparing datasets for training, validation, and testing.

    It supports multiple loading strategies and integrates optional pre-filtering and side information loading.
    The final output is a list of `DataSet` objects, ready to be consumed by the recommendation pipeline.

    Args:
        config (SimpleNamespace): Configuration namespace object defining data paths, splitting strategy,
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

    # TODO: insert separate reader config
    # TODO: move 'header' and 'columns' attributes to reader config

    config: DataLoadingConfig
    interactions: Union[list, pd.DataFrame]
    tuple_list: list
    side_information: SimpleNamespace

    def __init__(self, config: SimpleNamespace):
        self.logger = logging.get_logger(self.__class__.__name__)
        reader.logger = self.logger

        self.config = DataLoadingConfig(**vars(config.data_config))
        self.reader_config = self._get_reader_config()
        self.global_config = config

        # Default to align side information with the observed training set when present
        if self.config.side_information is not None and not hasattr(self.global_config, "align_side_with_train"):
            setattr(self.global_config, "align_side_with_train", True)

        if self.global_config.config_test:
            return

        self._load_ratings()
        self._load_side_information()
        self._preprocess_data()

    def _get_reader_config(self):
        return {
            "columns": self.config.columns,
            "datatypes": ["string", "string", "float", "float"],
            "sep": "\t",
            "ext": ".tsv",
            "header": self.config.header,
            "callback_fn": self._rename_cols_and_binarize
        }

    def _load_ratings(self):
        """Load user-item interaction data according to the selected strategy."""
        match self.config.strategy:

            case DataLoadingStrategy.FIXED:
                self.interactions = reader.read_tabular_split(
                    read_folder=self.config.data_folder,
                    **self.reader_config
                )

            case DataLoadingStrategy.HIERARCHY:
                self.interactions = reader.read_tabular_split(
                    read_folder=self.config.data_folder,
                    hierarchical=True,
                    **self.reader_config
                )

            case DataLoadingStrategy.DATASET:
                self.interactions = reader.read_tabular(
                    file_path=self.config.dataset_path,
                    **self.reader_config
                )

        self._clean(self._filter_nan_and_duplicates)

    def _rename_cols_and_binarize(self, data):
        names = ["userId", "itemId", "rating", "timestamp"]

        col_mapping = (
            {c: names[i] for i, c in enumerate(self.config.columns) if c in data.columns}
            if self.config.columns is not None
            else dict(zip(data.columns, names))
        )

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
        df = self.interactions

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
        sides = self.config.side_information
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

        if hasattr(self.global_config, "prefiltering"):
            prefilter = PreFilter(self.interactions, self.global_config.prefiltering)
            self.interactions = prefilter.filter()

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

        if isinstance(self.interactions, list):
            new_dataframe = []
            for tr, te in self.interactions:
                test = clean(te)
                train_fold = [(clean(tr_), clean(va)) for tr_, va in tr]
                new_dataframe.append((train_fold, test))
            self.interactions = new_dataframe
        else:
            self.interactions = clean(self.interactions)

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
        for side_obj in getattr(self, "_side_info_objs", []):
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

    def build(self) -> List[object]:
        if self.config.strategy != DataLoadingStrategy.DATASET:
            tuple_list = self.interactions
        else:
            self.logger.info("There will be the splitting")
            splitter = Splitter(self.interactions, self.global_config.splitting, self.config.seed)
            tuple_list = splitter.process_splitting()

        if len(tuple_list) > 1:
            self.logger.warning("You are using a splitting strategy with folds. "
                                "Paired TTest and Wilcoxon Test are not available!")
            self.global_config.evaluation.paired_ttest = False
            self.global_config.evaluation.wilcoxon_test = False

        data_list = []

        for p1, (train_val, test) in enumerate(tuple_list):
            # test level
            val_list = []
            for p2, (train, val) in enumerate(train_val):
                # validation level
                self.logger.info(
                    f"Test Fold {p1}{f" - Validation Fold {p2}" if val is not None else ""}"
                )
                single_data_object = DataSet(
                    config=self.global_config,
                    data_tuple=(train, val, test),
                    side_information_data=self.side_information
                )
                val_list.append(single_data_object)
            data_list.append(val_list)

        return data_list

    def generate_dataobjects_mock(self) -> List[object]:
        _column_names = ["userId", "itemId", "rating"]
        np.random.seed(self.config.seed)
        training_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))
        test_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))

        training_set = pd.DataFrame(np.array(training_set), columns=_column_names)
        test_set = pd.DataFrame(np.array(test_set), columns=_column_names)
        data_list = [[DataSet(self.global_config, (training_set, test_set))]]

        return data_list
