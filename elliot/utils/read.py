"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from ast import literal_eval

import torch
import pandas as pd
import configparser
import numpy as np
import os

from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from types import SimpleNamespace

from elliot.utils.folder import path_joiner, list_dir, is_dir, check_path
from elliot.utils.logging import get_logger


class Reader:
    def __init__(self, logger = get_logger("__main__")):
        self.logger = logger

    def read_tabular(
        self,
        file_path: str,
        columns: Optional[List[Union[str, int]]] = None,
        datatypes: Dict[Union[str, int], str] = {},
        sep: str = "\t",
        header: bool = False,
        callback_fn: Callable = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Read a tabular file and return a pandas DataFrame.

        The function supports column selection either by column names
        or by positional indices. If positional indices are provided,
        columns are selected and reordered accordingly.

        Args:
            file_path (str): Path to the input data file.
            columns (Optional[List[Union[str, int]]]):
                List of column names or positional indices.
                If integers are provided, columns are selected and reordered
                by position (e.g., [1, 2, 0]).
            datatypes (List[str]): List of data types to cast columns to.
            sep (str): Field separator used in the file.
            header (bool): Whether the file contains a header row.
            callback_fn (Callable): Optional function applied to the DataFrame.
            **kwargs (Any): Additional keyword arguments (unused).

        Returns:
            pd.DataFrame: Loaded and processed DataFrame.
        """
        try:
            # Determine header row index for pandas
            header_row = 0 if header else None

            data = pd.read_csv(file_path, sep=sep, header=header_row)
        except pd.errors.EmptyDataError:
            self.logger.warning(
                "The data file is empty. Returning an empty DataFrame."
            )
            # Create empty DataFrame with expected columns
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
                    df.columns = valid_idx

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

        self.logger.info(f"{file_path} - Loaded")

        if callback_fn is not None:
            df = callback_fn(df)

        return df

    def read_tabular_split(
        self,
        read_folder: str,
        ext: str = ".tsv",
        hierarchical: bool = False,
        **kwargs: Any
    ) -> List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]:

        tuple_list = []

        if not hierarchical:
            train_path = path_joiner(read_folder, f"train{ext}")
            test_path = path_joiner(read_folder, f"test{ext}")
            val_path = path_joiner(read_folder, f"val{ext}")

            train_df = self.read_tabular(train_path, **kwargs)
            test_df = self.read_tabular(test_path, **kwargs)

            if check_path(val_path):
                val_df = self.read_tabular(val_path, **kwargs)
            else:
                val_df = None

            tuple_list = [([(train_df, val_df)], test_df)]

        else:
            test_dirs = [p for p in list_dir(read_folder) if is_dir(p)]

            for test_folder_path in test_dirs:
                test_path = path_joiner(test_folder_path, f"test{ext}")

                test_df = self.read_tabular(test_path, **kwargs)

                val_dirs = [p for p in list_dir(test_folder_path) if is_dir(p)]
                val_list = []

                for val_folder_path in val_dirs:
                    train_path = path_joiner(val_folder_path, f"train{ext}")
                    val_path = path_joiner(val_folder_path, f"val{ext}")

                    train_df = self.read_tabular(train_path, **kwargs)
                    val_df = self.read_tabular(val_path, **kwargs)

                    val_list.append((train_df, val_df))

                if not val_list:
                    train_path = path_joiner(test_folder_path, f"train{ext}")

                    train_df = self.read_tabular(train_path, **kwargs)

                    val_list.append((train_df, None))

                tuple_list.append((val_list, test_df))

        return tuple_list

    def read_negatives(
        self,
        read_folder: str,
        sep: str = "\t",
        ext: str = ".tsv",
        scope: str = "test",
        **kwargs: Any
    ) -> Dict[str, List[str]]:

        file_path = path_joiner(read_folder, f"{scope}_negative{ext}")
        neg = {}

        with open(file_path) as file:
            for line in file:
                line = line.rstrip("\n").split(sep)
                user_id = str(literal_eval(line[0])[0])
                neg[user_id] = [i for i in line[1:]]

        return neg

    def read_model(
        self,
        read_folder: str,
        model_name: str
    ) -> Any:

        file_path = path_joiner(read_folder, model_name, f"best-weights-{model_name}.pth")
        model = torch.load(file_path)

        self.logger.info(
            "Model restored from disk",
            extra={"context": {"path": file_path}}
        )

        return model


def read_csv(filename):
    """
    Args:
        filename (str): csv file path
    Return:
         A pandas dataframe.
    """
    df = pd.read_csv(filename, index_col=False)
    return df


def read_np(filename):
    """
    Args:
        filename (str): filename of numpy to load
    Return:
        The loaded numpy.
    """
    return np.load(filename)


def read_imagenet_classes_txt(filename):
    """
    Args:
        filename (str): txt file path
    Return:
         A list with 1000 imagenet classes as strings.
    """
    with open(filename) as f:
        idx2label = eval(f.read())

    return idx2label


def read_config(sections_fields):
    """
    Args:
        sections_fields (list): list of fields to retrieve from configuration file
    Return:
         A list of configuration values.
    """
    config = configparser.ConfigParser()
    config.read('./config/configs.ini')
    configs = []
    for s, f in sections_fields:
        configs.append(config[s][f])
    return configs


def read_multi_config():
    """
    It reads a config file that contains the configuration parameters for the recommendation systems.

    Return:
         A list of configuration settings.
    """
    config = configparser.ConfigParser()
    config.read('./config/multi.ini')
    configs = []
    for section in config.sections():
        single_config = SimpleNamespace()
        single_config.name = section
        for field, value in config.items(section):
            single_config.field = value
        configs.append(single_config)
    return configs



def find_checkpoint(dir, restore_epochs, epochs, rec, best=0):
    """
    :param dir: directory of the model where we start from the reading.
    :param restore_epochs: epoch from which we start from.
    :param epochs: epochs from which we restore (0 means that we have best)
    :param rec: recommender model
    :param best: 0 No Best - 1 Search for the Best
    :return:
    """
    if best:
        for r, d, f in os.walk(dir):
            for file in f:
                if 'best-weights-'.format(restore_epochs) in file:
                    return dir + file.split('.')[0]
        return ''

    if rec == "apr" and restore_epochs < epochs:
        # We have to restore from an execution of bprmf
        dir_stored_models = os.walk('/'.join(dir.split('/')[:-2]))
        for dir_stored_model in dir_stored_models:
            if 'bprmf' in dir_stored_model[0]:
                dir = dir_stored_model[0] + '/'
                break

    for r, d, f in os.walk(dir):
        for file in f:
            if 'weights-{0}-'.format(restore_epochs) in file:
                return dir + file.split('.')[0]
    return ''
