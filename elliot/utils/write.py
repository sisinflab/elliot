"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle
import os
import numpy as np
import pandas as pd

from elliot.utils.folder import create_folder


def save_tabular_df(
    df: pd.DataFrame,
    folder_path: str,
    filename: str,
    sep: str = "\t"
):
    """Save a DataFrame in tabular format to the specified path.

    Args:
        df (pd.Dataframe): Dataframe to save.
        folder_path (str): Destination folder.
        filename (str): Name of the destination file (e.g., 'val.tsv').
        sep (str): Separator to use, default is `\\t`.
    """
    # Create the folder if it does not exist
    create_folder(folder_path, exist_ok=True)

    # Build the full file path
    file_path = os.path.abspath(os.path.join(folder_path, filename))

    # Save as tabular data
    df.to_csv(
        file_path,
        sep=sep,
        index=False,
        header=False
    )


def save_obj(obj, name):
    """
    Store the object in a pkl file
    :param obj: python object to be stored
    :param name: file name (Not insert .pkl)
    :return:
    """
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f)


def save_np(npy, filename):
    """
    Store numpy to memory.
    Args:
        npy: numpy to save
        filename (str): filename
    """
    np.save(filename, npy)


def store_recommendation(recommendations, path=""):
    """
    Store recommendation list (top-k)
    :return:
    """

    with open(path, 'w') as out:
        for u, recs in recommendations.items():
            for i, value in recs:
                out.write(str(u) + '\t' + str(i) + '\t' + str(value) + '\n')
