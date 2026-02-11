"""
Module description:

"""

from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from ast import literal_eval
from logging import LoggerAdapter
import fnmatch
import torch
import pandas as pd
import configparser
import numpy as np
import os
from types import SimpleNamespace

from elliot.utils.folder import path_joiner, list_dir, is_dir, is_file, file_ext, file_name
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
        **kwargs: Any
    ) -> List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]:
        """Read tabular data splits from a specified folder,
        supporting both classic and hierarchical split structures.

        Args:
            read_folder (str): Path to the folder containing tabular data files or other folders
                for hierarchical splits.
            hierarchical (bool, optional): Whether the data follows a hierarchical
                split structure. Defaults to False.
            **kwargs (Any): Additional keyword arguments passed to `read_folder` and `read_tabular` methods.

        Returns:
            List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]:
                A list of tuples where each tuple contains a list of train/validation
                DataFrame pairs and a test DataFrame.
        """

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

            train_df = self.read_tabular(train_path, **kwargs)
            test_df = self.read_tabular(test_path, **kwargs)
            val_df = self.read_tabular(val_path, **kwargs) if val_path is not None else None

            tuple_list = [([(train_df, val_df)], test_df)]

        # Case 2: hierarchical split
        else:
            test_dirs = [p for p in list_dir(read_folder) if is_dir(p)]

            # Test fold level
            for test_folder_path in test_dirs:
                test_path = get_file_path(test_folder_path, "test")

                test_df = self.read_tabular(test_path, **kwargs)

                val_dirs = [p for p in list_dir(test_folder_path) if is_dir(p)]
                val_list = []

                # Validation fold level
                for val_folder_path in val_dirs:
                    train_path = get_file_path(val_folder_path, "train")
                    val_path = get_file_path(val_folder_path, "val")

                    train_df = self.read_tabular(train_path, **kwargs)
                    val_df = self.read_tabular(val_path, **kwargs)

                    val_list.append((train_df, val_df))

                # Load only train dataset if validation folds are missing
                if not val_list:
                    train_path = get_file_path(test_folder_path, "train")

                    train_df = self.read_tabular(train_path, **kwargs)

                    val_list.append((train_df, None))

                tuple_list.append((val_list, test_df))

        return tuple_list

    def read_negatives(
        self,
        read_folder: str,
        sep: str = "\t",
        scope: str = "test",
        **kwargs: Any
    ) -> Dict[str, List[str]]:
        """Read negative samples from a specified folder and return them as a dictionary.

        Args:
            read_folder (str): Path to the folder containing the negative samples file.
            sep (str): Field separator used in the input file. Defaults to "\\t".
            scope (str): Scope name to construct the file name. Defaults to "test".
            **kwargs (Any): Additional keyword arguments passed to the `read_folder` method.

        Returns:
            Dict[str, List[str]]: A dictionary mapping user IDs to lists of negative samples.

        Raises:
            FileNotFoundError: If the specified file does not exist in the given folder.
            ValueError: If the file content cannot be parsed as expected.
        """
        files = self.read_folder(read_folder, **kwargs)
        by_name = {file_name(p): p for p in files}
        path = by_name.get(f"{scope}_negative")

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
        files = self.read_folder(path_joiner(read_folder, model_name), **kwargs)
        by_name = {file_name(p): p for p in files}
        path = by_name.get(f"best-weights-{model_name}")

        model = torch.load(path)

        self.logger.info(
            "Model restored from disk",
            extra={"context": {"path": path}}
        )

        return model


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
