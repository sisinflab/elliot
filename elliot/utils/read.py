"""
Module description:

"""
import json
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from ast import literal_eval
from logging import LoggerAdapter
import fnmatch
import csv
import torch
import pandas as pd
import configparser
import numpy as np
import os
from types import SimpleNamespace

from elliot.utils.folder import list_dir, is_dir, is_file, file_ext, file_name
from elliot.utils.logging import get_logger


class Reader:
    """Utility class for reading and processing various types of data files.

    Attributes:
        logger (LoggerAdapter): A logging instance.
    """

    def __init__(self, logger: LoggerAdapter = get_logger("__main__")):
        self.logger = logger

    def read_tabular(
        self,
        path: str,
        header: bool = False,
        columns: Optional[List[Union[str, int]]] = None,
        datatypes: Dict[Union[str, int], str] = {},
        sep: str = "\t",
        callback_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read tabular data from a file and return it as a pandas DataFrame,
        handling variations in columns and data types.

        Args:
            path (str): Path to the file containing the tabular data.
            header (bool): Whether the input file contains a header row. Defaults to False.
            columns (List[Union[str, int]], optional): List of column names or indices
                to select. Defaults to None.
            datatypes (Dict[Union[str, int], str], optional): Mapping of column names or indices
                to data types. Defaults to {}.
            sep (str, optional): Column separator used in the input file. Defaults to "\\t".
            callback_fn (Callable, optional): Function to apply to the resulting DataFrame
                before returning. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to the `callback_fn` function.

        Returns:
            pd.DataFrame: A pandas DataFrame containing the loaded tabular data.
        """
        try:
            # Determine header row index for pandas
            header_row = 0 if header else None

            data = pd.read_csv(path, sep=sep, header=header_row)
        except pd.errors.EmptyDataError:
            self.logger.warning(
                "The data file is empty. Returning an empty DataFrame."
            )
            # Create an empty DataFrame with expected columns
            cols = (
                columns if columns and all(isinstance(c, str) for c in columns)
                else []
            )
            df = pd.DataFrame(columns=cols)

            # Apply dtypes if provided
            if datatypes:
                df = df.astype({c: t for c, t in datatypes.items() if c in df.columns})
        else:
            # Check whether columns are specified as positional indices
            is_positional = columns is not None and any(isinstance(c, int) for c in columns)

            # Assign column names only if header is missing and columns are semantic
            if not header and columns is not None and not is_positional:
                data.columns = columns[:len(data.columns)]

            # Case 1: no column selection requested
            if columns is None:
                df = data

            # Case 2: positional column selection and reordering
            elif is_positional:
                columns = [c for c in columns if isinstance(c, int)]
                max_idx = data.shape[1] - 1
                valid_idx = [i for i in columns if 0 <= i <= max_idx]

                if not valid_idx:
                    self.logger.warning(
                        "None of the desired column indices were found. Returning an empty DataFrame."
                    )
                    df = pd.DataFrame()
                else:
                    df = data.iloc[:, valid_idx]

                    # Apply datatypes if provided
                    dtype_to_use = {
                        df.columns[i]: datatypes[i]
                        for i in range(len(valid_idx)) if i in datatypes
                    }
                    df = df.astype(dtype_to_use)

            # Case 3: semantic column selection by name
            else:
                cols_to_use = [c for c in columns if c in data.columns]
                if not cols_to_use:
                    self.logger.warning(
                        "None of the desired columns were found. Returning an empty DataFrame."
                    )
                    df = pd.DataFrame()
                else:
                    # Apply datatypes if provided
                    dtype_to_use = {c: d for c, d in datatypes.items() if c in cols_to_use}
                    df = data[cols_to_use].astype(dtype_to_use)

        self.logger.info(f"Loaded: {path}")

        if callback_fn is not None:
            df = callback_fn(df, **kwargs)

        return df

    def read_sequence_tabular(
        self,
        path: str,
        format: str = "wide",
        header: bool = True,
        columns: Optional[List[Union[str, int]]] = None,
        datatypes: Dict[Union[str, int], str] = {},
        sequence_sep: str = " ",
        sep: str = "\t",
        callback_fn: Optional[Callable] = None,
        track_source_rows: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read sequential interaction data from a file and return it as a long-format
        pandas DataFrame with one row per (user, item) interaction, handling variations
        in columns and data types.

        Two on-disk layouts are supported through the `format` argument:

        - "wide": each row holds a ragged, `sep`-separated sequence where the first
            token is the user identifier and the remaining tokens are the interacted
            items (e.g. "u0\\titem1\\titem2\\titem3"). Only the first entry of `columns`
            is used, to name the user column.
        - "inline": each row holds a user identifier plus a single column containing
            the whole interaction sequence serialized as a string
            (e.g. "70,\\"495 1631 2317\\""), with the selected columns interpreted,
            in order, as [user, sequence, timestamp (optional), *metadata (optional)].

        A given user identifier may span several rows in the source file, each one
        representing a distinct session for that user.

        Args:
            path (str): Path to the file containing the sequential data.
            format (str): Layout of the input file, either "wide" or "inline". Defaults to "wide".
            header (bool): Whether the input file contains a header row. Defaults to True.
            columns (List[Union[str, int]], optional): List of column names or indices
                to select. Defaults to None.
            datatypes (Dict[Union[str, int], str], optional): Mapping of column names or indices
                to data types. Defaults to {}.
            sequence_sep (str): Separator used inside the sequence string. Only used when
                `format` is "inline". Defaults to " ".
            sep (str, optional): Column/token separator used in the input file. Defaults to "\\t".
            callback_fn (Callable, optional): Function to apply to the resulting DataFrame
                before returning. Defaults to None.
            track_source_rows (bool): If True, add a "_sourceRow" column recording, for
                every exploded item, the position of the raw source row (i.e. session) it
                came from. Defaults to False.
            **kwargs (Any): Additional keyword arguments passed to the `callback_fn` function.

        Returns:
            pd.DataFrame: A long-format pandas DataFrame with one row per (user, itemId)
                interaction, plus timestamp/meta columns when available and `format`
                is "inline".

        Raises:
            ValueError: If `format` is not one of "wide" or "inline".
        """
        item_col = "itemId"

        # Case "inline": one row per session, sequence serialized as a string
        if format == "inline":
            data = self.read_tabular(path, header=header, columns=columns, datatypes=datatypes, sep=sep)
            result_cols = list(data.columns)

            # Not enough columns to identify the user and the sequence
            if len(result_cols) < 2:
                self.logger.warning(
                    "The user or sequence column was not found. Returning an empty DataFrame."
                )
                user_col = columns[0] if columns and isinstance(columns[0], str) else "userId"
                return pd.DataFrame(columns=[user_col, item_col])

            # Interpret columns, in order, as [user, sequence, timestamp (optional), *metadata (optional)]
            user_col, sequence_col = result_cols[0], result_cols[1]
            timestamp_col = result_cols[2] if len(result_cols) > 2 else None
            meta_cols = result_cols[3:]

            cols_to_keep = [user_col, sequence_col]
            if timestamp_col is not None:
                cols_to_keep.append(timestamp_col)
            cols_to_keep.extend(meta_cols)

            # Drop rows with missing values before exploding the sequence
            data = data[cols_to_keep].dropna()

            # Record the source row (session) for every exploded item
            if track_source_rows:
                data["_sourceRow"] = np.arange(len(data))
            
            # Split the serialized sequence into individual item tokens
            data[item_col] = data[sequence_col].astype(str).str.split(sequence_sep)
            
            # One row per (user, item) interaction
            data = data.explode(item_col)
            
            # Drop the now-redundant serialized sequence column
            data = data.drop(columns=[sequence_col])
            data[item_col] = data[item_col].str.strip()
            
            # Remove empty tokens produced by trailing separators
            df = data[data[item_col] != ""].reset_index(drop=True)

        # Case "wide": ragged, sep-separated line per session
        elif format == "wide":
            user_col = columns[0] if columns else 0

            # Determine header row index for pandas
            header_row = 0 if header else None

            # Read line-by-line to preserve the ragged rows as raw sep-separated tokens
            raw = pd.read_csv(
                path,
                sep="\0",
                header=header_row,
                names=["_raw"],
                quoting=csv.QUOTE_NONE,
            )

            if raw.empty:
                self.logger.warning(
                    "The data file is empty. Returning an empty DataFrame."
                )
                return pd.DataFrame(columns=[user_col, item_col])

            # Split each line into tokens: the first is the user identifier, the rest are items
            tokens = raw["_raw"].str.split(sep)
            data = pd.DataFrame({
                user_col: tokens.str[0].str.strip(),
                item_col: tokens.str[1:],
            })
            
            # Record the source row (session) for every exploded item
            if track_source_rows:
                data["_sourceRow"] = np.arange(len(data))
            
            # One row per (user, item) interaction
            data = data.explode(item_col).dropna(subset=[item_col])
            data[item_col] = data[item_col].astype(str).str.strip()
            
            # Remove empty tokens produced by trailing separators
            df = data[data[item_col] != ""].reset_index(drop=True)

            self.logger.info(f"Loaded: {path}")

        else:
            raise ValueError(f"Unsupported format '{format}'. Expected 'wide' or 'inline'.")

        # Apply datatypes if provided
        if datatypes:
            dtype_to_use = {c: d for c, d in datatypes.items() if c in df.columns}
            df = df.astype(dtype_to_use)

        if callback_fn is not None:
            df = callback_fn(df, **kwargs)

        return df

    def read_folder(
        self,
        folder: str,
        patterns: Optional[str] = None,
        ext: Optional[List[str]] = None,
        **kwargs: Any
    ) -> List[str]:
        """Get all files in a folder with optional filtering by patterns and extensions.

        Args:
            folder (str): Path to the folder to read.
            patterns (str, optional): Filename patterns to match (e.g., "*.tsv"). Defaults to None.
            ext (List[str], optional): File extension(s) to filter by (e.g., ".tsv"). Defaults to None.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            List[str]: List of filenames in the folder that match the specified patterns and/or extensions.
        """
        # Get all the files in the folder
        only_files = [f for f in list_dir(folder) if is_file(f)]

        # Optionally filter files by filename patterns (e.g., "*.tsv")
        if patterns is not None:
            patterns = patterns if isinstance(patterns, list) else [patterns]
            only_files = [f for f in only_files if any(fnmatch.fnmatch(f, p) for p in patterns)]

        # Optionally filter files by extension (e.g., ".tsv")
        if ext is not None:
            only_files = [f for f in only_files if file_ext(f).lower() in ext]

        return only_files

    def read_tabular_split(
        self,
        read_folder: str,
        hierarchical: bool = False,
        sequential: bool = False,
        **kwargs: Any
    ) -> List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], Optional[pd.DataFrame], pd.DataFrame]]:
        """Read tabular data splits from a specified folder,
        supporting both classic and hierarchical split structures.

        Args:
            read_folder (str): Path to the folder containing tabular data files or other folders
                for hierarchical splits.
            hierarchical (bool, optional): Whether the data follows a hierarchical
                split structure. Defaults to False.
            sequential (bool, optional): Whether each split file stores sequential interaction
                data (see `read_sequence_tabular`) instead of plain tabular data. Defaults to False.
            **kwargs (Any): Additional keyword arguments passed to `read_folder` and to
                `read_sequence_tabular` (if `sequential` is True) or `read_tabular` (otherwise).

        Returns:
            List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], Optional[pd.DataFrame], pd.DataFrame]]:
                A list of tuples where each tuple contains an optional list of train/validation
                DataFrame pairs, a train DataFrame, and a test DataFrame.
        """
        read_fn = self.read_sequence_tabular if sequential else self.read_tabular

        def get_file_path(folder, name):
            files = self.read_folder(folder, **kwargs)
            by_name = {file_name(p): p for p in files}
            return by_name.get(name)

        tuple_list = []

        # Case 1: classic train, val, test split
        if not hierarchical:
            train_path = get_file_path(read_folder, "train")
            test_path = get_file_path(read_folder, "test")
            val_path = get_file_path(read_folder, "val")

            train_df = read_fn(train_path, **kwargs)
            test_df = read_fn(test_path, **kwargs)

            if val_path is not None:
                val_df = read_fn(val_path, **kwargs)
                folds = [(train_df, val_df)]
                original_train_df = None
            else:
                folds = []
                original_train_df = train_df

            tuple_list = [(folds, original_train_df, test_df)]

        # Case 2: hierarchical split
        else:
            test_dirs = [p for p in list_dir(read_folder) if is_dir(p)]

            # Test fold level
            for test_folder_path in test_dirs:
                test_path = get_file_path(test_folder_path, "test")

                test_df = read_fn(test_path, **kwargs)

                val_dirs = [p for p in list_dir(test_folder_path) if is_dir(p)]
                val_list = []
                original_train_df = None

                # Validation fold level
                for val_folder_path in val_dirs:
                    train_path = get_file_path(val_folder_path, "train")
                    val_path = get_file_path(val_folder_path, "val")

                    train_df = read_fn(train_path, **kwargs)
                    val_df = read_fn(val_path, **kwargs)

                    val_list.append((train_df, val_df))

                if val_list:
                    train_df, val_df = val_list[0]
                    if len(val_list) > 1:
                        original_train_df = pd.concat([train_df, val_df], ignore_index=True)

                # Load only train dataset if validation folds are missing
                else:
                    train_path = get_file_path(test_folder_path, "train")
                    original_train_df = read_fn(train_path, **kwargs)

                tuple_list.append((val_list, original_train_df, test_df))

        return tuple_list

    def read_mapping(
        self,
        path: str,
        sep: str = "\t",
        dtype: str = "str",
        remove_duplicates: bool = True,
        callback_fn: Optional[Callable] = None,
        **kwargs: Any,
    ) -> Dict[Any, List[Any]]:

        DTYPES = {"int": int, "float": float, "str": str}
        dtype = DTYPES[dtype]

        mapping = {}

        with open(path) as file:
            for raw in file:
                parts = raw.rstrip("\n").split(sep)

                head = parts[0]
                if isinstance(head, str):
                    head = literal_eval(head)
                if isinstance(head, list):
                    head = head[0]

                identifier = dtype(head)
                mapping[identifier] = [dtype(x) for x in parts[1:]]

                if remove_duplicates:
                    mapping[identifier] = list(set(mapping[identifier]))

        self.logger.info(f"Loaded: {path}")

        if callback_fn is not None:
            mapping = callback_fn(mapping, **kwargs)

        return mapping

    def read_negatives(
        self,
        read_folder: str,
        sep: str = "\t",
        fold_index: Tuple[int, Optional[int]] = (0, None),
        **kwargs: Any
    ) -> Dict[str, List[str]]:
        """Read negative samples from a specified folder and return them as a dictionary.

        Args:
            read_folder (str): Path to the folder containing the negative samples file.
            sep (str): Field separator used in the input file. Defaults to "\\t".
            fold_index (Tuple[int, Optional[int]]): Tuple containing the complete fold index.
            **kwargs (Any): Additional keyword arguments passed to the `read_folder` method.

        Returns:
            Dict[str, List[str]]: A dictionary mapping user IDs to lists of negative samples.

        Raises:
            FileNotFoundError: If the specified file does not exist in the given folder.
            ValueError: If the file content cannot be parsed as expected.
        """
        files = self.read_folder(read_folder, **kwargs)
        by_name = {file_name(p): p for p in files}

        suffix = f"_val{fold_index[1] + 1}" if fold_index[1] is not None else ""
        name = f"test{fold_index[0] + 1}{suffix}_negative"
        path = by_name.get(name)

        neg = {}

        with open(path) as file:
            for line in file:
                line = line.rstrip("\n").split(sep)
                user_id = str(literal_eval(line[0])[0])
                neg[user_id] = [i for i in line[1:]]

        self.logger.info(f"Loaded: {path}")

        return neg

    def read_model(
        self,
        read_folder: str,
        model_name: str,
        **kwargs: Any
    ) -> Any:
        """Read a model from the specified folder and with the provided name.

        Args:
            read_folder (str): Path to the folder containing the model.
            model_name (str): The name of the model whose weights need to be loaded.
            **kwargs (Any): Additional keyword arguments passed to the `read_folder` method.

        Returns:
            Any: The loaded model object with its weights restored.

        Raises:
            FileNotFoundError: If the file is not found in the specified folder.
            RuntimeError: If there is an error while loading the model.
        """
        files = self.read_folder(read_folder, **kwargs)
        by_name = {file_name(p): p for p in files}
        path = by_name.get(f"best-weights-{model_name}")

        model = torch.load(path)

        self.logger.info(
            "Model restored from disk",
            extra={"context": {"path": path}}
        )

        return model

    def read_json(
        self,
        path: str,
        **kwargs: Any
    ) -> Any:
        """Read and parse data from a JSON file.

        Args:
            path (str): Path to the JSON file.
            **kwargs (Any): Additional keyword arguments.

        Returns:
            Any: The loaded and parsed JSON data.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.logger.info(f"Loaded: {path}")

        return data

# def read_csv(filename):
#     """
#     Args:
#         filename (str): csv file path
#     Return:
#          A pandas dataframe.
#     """
#     df = pd.read_csv(filename, index_col=False)
#     return df
#
#
# def read_np(filename):
#     """
#     Args:
#         filename (str): filename of numpy to load
#     Return:
#         The loaded numpy.
#     """
#     return np.load(filename)
#
#
# def read_imagenet_classes_txt(filename):
#     """
#     Args:
#         filename (str): txt file path
#     Return:
#          A list with 1000 imagenet classes as strings.
#     """
#     with open(filename) as f:
#         idx2label = eval(f.read())
#
#     return idx2label
#
#
# def read_config(sections_fields):
#     """
#     Args:
#         sections_fields (list): list of fields to retrieve from configuration file
#     Return:
#          A list of configuration values.
#     """
#     config = configparser.ConfigParser()
#     config.read('./config/configs.ini')
#     configs = []
#     for s, f in sections_fields:
#         configs.append(config[s][f])
#     return configs
#
#
# def read_multi_config():
#     """
#     It reads a config file that contains the configuration parameters for the recommendation systems.
#
#     Return:
#          A list of configuration settings.
#     """
#     config = configparser.ConfigParser()
#     config.read('./config/multi.ini')
#     configs = []
#     for section in config.sections():
#         single_config = SimpleNamespace()
#         single_config.name = section
#         for field, value in config.items(section):
#             single_config.field = value
#         configs.append(single_config)
#     return configs
#
#
#
# def find_checkpoint(dir, restore_epochs, epochs, rec, best=0):
#     """
#     :param dir: directory of the model where we start from the reading.
#     :param restore_epochs: epoch from which we start from.
#     :param epochs: epochs from which we restore (0 means that we have best)
#     :param rec: recommender model
#     :param best: 0 No Best - 1 Search for the Best
#     :return:
#     """
#     if best:
#         for r, d, f in os.walk(dir):
#             for file in f:
#                 if 'best-weights-'.format(restore_epochs) in file:
#                     return dir + file.split('.')[0]
#         return ''
#
#     if rec == "apr" and restore_epochs < epochs:
#         # We have to restore from an execution of bprmf
#         dir_stored_models = os.walk('/'.join(dir.split('/')[:-2]))
#         for dir_stored_model in dir_stored_models:
#             if 'bprmf' in dir_stored_model[0]:
#                 dir = dir_stored_model[0] + '/'
#                 break
#
#     for r, d, f in os.walk(dir):
#         for file in f:
#             if 'weights-{0}-'.format(restore_epochs) in file:
#                 return dir + file.split('.')[0]
#     return ''
