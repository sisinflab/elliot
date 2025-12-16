from typing import List, Optional
from types import SimpleNamespace
import os
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
from elliot.utils.read import read_tabular
from elliot.utils.validation import DataLoadingConfig


class DataSetLoader:
    """The DataSetLoader class is responsible for loading and preparing datasets for training, validation, and testing.

    It supports multiple loading strategies and integrates optional pre-filtering and side information loading.
    The final output is a list of `DataSet` objects, ready to be consumed by the recommendation pipeline.

    Args:
        config_ns (SimpleNamespace): Configuration namespace object defining data paths, splitting strategy,
            filters, etc.
        *args (tuple): Additional positional arguments.
        **kwargs (dict): Additional keyword arguments.

    Supported Loading Strategies:

    - `fixed`: Load train/test/(optional) validation sets from files.
    - `hierarchy`: Load multiple folds from a nested directory structure.
    - `dataset`: Load a single dataset and later applies pre-filtering and splitting.

    To configure the data loading, include the appropriate
    settings in the configuration file using the pattern shown below.

    .. code:: yaml

      data_config:
        strategy: dataset|fixed|hierarchy
        header: True|False
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

    strategy: DataLoadingStrategy
    dataset_path: Optional[str] = None
    root_folder: Optional[str] = None
    train_path: Optional[str] = None
    validation_path: Optional[str] = None
    test_path: Optional[str] = None
    header: bool = False
    binarize: bool = False
    side_information: Optional[SimpleNamespace] = None
    seed: int = 42

    def __init__(self, config_ns: SimpleNamespace, *args, **kwargs):
        self.logger = logging.get_logger(self.__class__.__name__)
        self.args = args
        self.kwargs = kwargs
        self.config_ns = config_ns
        self.dataset_loading_ns = config_ns.data_config
        self.dataframe = None
        self.tuple_list = None

        self.set_params()

        # Default to aligning side information with the observed training set when present
        if getattr(self.dataset_loading_ns, "side_information", None) and not hasattr(self.config_ns, "align_side_with_train"):
            setattr(self.config_ns, "align_side_with_train", True)

        if self.config_ns.config_test:
            return

        self._load_ratings()
        self._load_side_information()
        self._preprocess_data()

        if isinstance(self.tuple_list[0][1], list):
            self.logger.warning("You are using a splitting strategy with folds. "
                                "Paired TTest and Wilcoxon Test are not available!")
            self.config_ns.evaluation.paired_ttest = False
            self.config_ns.evaluation.wilcoxon_test = False

    def set_params(self):
        """Validate and set object parameters."""
        config = DataLoadingConfig(**vars(self.dataset_loading_ns))

        for name, val in config.get_validated_params().items():
            setattr(self, name, val)

        self.binarize = self.config_ns.binarize

    def _load_ratings(self):
        """Load user-item interaction data according to the selected strategy."""
        match self.strategy:

            case DataLoadingStrategy.FIXED:
                train_df = self._read_from_tsv(self.train_path)
                self.logger.info(f"{self.train_path} - Loaded")

                test_df = self._read_from_tsv(self.test_path)
                self.logger.info(f"{self.test_path} - Loaded")

                if self.validation_path is not None:
                    val_df = self._read_from_tsv(self.validation_path)
                    self.logger.info(f"{self.validation_path} - Loaded")

                    self.dataframe = [([(train_df, val_df)], test_df)]
                else:
                    self.dataframe = [(train_df, test_df)]

            case DataLoadingStrategy.HIERARCHY:
                self.dataframe = self._read_splitting(self.root_folder)
                self.logger.info(f"{self.root_folder} - Loaded splitting")

            case DataLoadingStrategy.DATASET:
                self.dataframe = self._read_from_tsv(self.dataset_path)
                self.logger.info(f"{self.dataset_path} - Loaded")

    def _read_from_tsv(self, file_path: str) -> pd.DataFrame:
        """Load a TSV file, process the timestamp and optionally binarize ratings.

        Args:
            file_path (str): Path to the TSV file containing interactions.

        Returns:
            pd.DataFrame: The loaded and formatted DataFrame.
        """

        cols = ['userId', 'itemId', 'rating', 'timestamp']
        dtypes = ['str', 'str', 'float', 'float']

        df = read_tabular(
            file_path,
            cols=cols,
            datatypes=dtypes,
            sep='\t',
            header=self.header
        )

        if self.binarize == True or 'rating' not in cols:
            df["rating"] = 1.0

        return df

    def _read_splitting(self, folder_path: str) -> list:
        """Read train/val/test splits organized in a hierarchical folder structure.

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
        """Load side information (e.g., user/item features) using custom dataloaders defined in config.

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
        sides = self.dataset_loading_ns.side_information
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
        Perform optional pre-filtering, and dataset splitting, only if the "dataset" strategy is used.
        """
        self._intersect_users_items()
        self._clean()
        self._maybe_materialize_cache()

        del self._items, self._users, self._side_info_objs

        if self.strategy != DataLoadingStrategy.DATASET:
            self.tuple_list = self.dataframe
            return

        if hasattr(self.config_ns, 'prefiltering'):
            prefilter = PreFilter(self.dataframe, self.config_ns.prefiltering)
            self.dataframe = prefilter.filter()

        self.logger.info("There will be the splitting")
        splitter = Splitter(self.dataframe, self.config_ns.splitting, self.seed)
        self.tuple_list = splitter.process_splitting()

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

    def _clean(self):
        """Clean all loaded DataFrames by filtering users/items and removing duplicates."""
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

    def _clean_single_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter a single DataFrame based on valid users/items and applies basic cleanup,
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

    def generate_dataobjects(self) -> List[object]:
        data_list = []
        for p1, (train_val, test) in enumerate(self.tuple_list):
            # testset level
            if isinstance(train_val, list):
                # validation level
                val_list = []
                for p2, (train, val) in enumerate(train_val):
                    self.logger.info(f"Test Fold {p1} - Validation Fold {p2}")
                    single_dataobject = DataSet(self.config_ns, (train, val, test), self.side_information, self.args,
                                                self.kwargs)
                    val_list.append(single_dataobject)
                data_list.append(val_list)
            else:
                self.logger.info(f"Test Fold {p1}")
                single_dataobject = DataSet(self.config_ns, (train_val, test), self.side_information, self.args,
                                            self.kwargs)
                data_list.append([single_dataobject])
        return data_list

    def generate_dataobjects_mock(self) -> List[object]:
        np.random.seed(self.seed)
        _column_names = ['userId', 'itemId', 'rating']
        training_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))
        test_set = np.hstack(
            (np.random.randint(0, 5 * 20, size=(5 * 20, 2)), np.random.randint(0, 2, size=(5 * 20, 1))))

        training_set = pd.DataFrame(np.array(training_set), columns=_column_names)
        test_set = pd.DataFrame(np.array(test_set), columns=_column_names)
        data_list = [[DataSet(self.config_ns, (training_set, test_set), self.args, self.kwargs)]]

        return data_list
