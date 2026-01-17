"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from ast import literal_eval
from collections import defaultdict

import torch
import pandas as pd
import configparser
import numpy as np
import os

from typing import List, Tuple, Dict, Any, Callable, Optional
from types import SimpleNamespace

from elliot.utils.folder import path_joiner, list_dir, is_dir, check_path
from elliot.utils.logging import get_logger


class Reader:
    def __init__(self, logger = get_logger("__main__")):
        self.logger = logger

    def read_tabular(
        self,
        file_path: str,
        columns: Optional[List[str]] = None,
        datatypes: List[str] = [],
        sep: str = "\t",
        header: bool = False,
        callback_fn: Callable = None,
        **kwargs: Any,
    ) -> pd.DataFrame:

        try:
            header_row = 0 if header else None
            data = pd.read_csv(file_path, sep=sep, header=header_row)
        except pd.errors.EmptyDataError:
            self.logger.warning(
                "The data file is empty. Returning an empty DataFrame."
            )
            cols_to_use = columns if columns is not None else []
            dtype_to_use = {c: d for c, d in zip(cols_to_use, datatypes)}
            df = pd.DataFrame(columns=cols_to_use).astype(dtype_to_use)
        else:
            if not header and columns is not None:
                data.columns = columns[:len(data.columns)]

            if columns is None:
                dtype_to_use = {c: d for c, d in zip(list(data.columns), datatypes)}
                df = data.astype(dtype_to_use)
            else:
                cols_to_use = [c for c in columns if c in data.columns]
                if not cols_to_use:
                    self.logger.warning(
                        "None of the desired columns were found. Returning an empty DataFrame."
                    )
                    df = pd.DataFrame()
                else:
                    dtype_to_use = {
                        c: datatypes[i]
                        for i, c in enumerate(columns)
                        if datatypes and c in data.columns
                    }
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
