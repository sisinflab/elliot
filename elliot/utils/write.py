"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from typing import List, Tuple, Dict, Any, Callable, Optional
import torch
import numpy as np
import pandas as pd

from elliot.utils.folder import check_dir, path_joiner
from elliot.utils.logging import get_logger


class Writer:
    def __init__(self, logger = get_logger("__main__")):
        self.logger = logger

    def write_tabular(
        self,
        df: pd.DataFrame,
        file_path: str,
        sep: str = "\t",
        callback_fn: Callable = None,
        **kwargs: Any
    ):
        df.to_csv(
            file_path,
            sep=sep,
            index=False,
            header=False
        )

        if callback_fn is not None:
            callback_fn()

    def write_split(
        self,
        fold_dataset: List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]],
        save_folder: str,
        ext: str = ".tsv",
        **kwargs: Any
    ):
        """Write the generated splits to disk.

        Args:
            fold_dataset (List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]):
                A list of split tuples to be saved on disk.
            save_folder (str): Destination folder.
            ext (str): File extension to use; default is `.tsv`.
        """
        check_dir(save_folder, replace=True)

        for i, (train_val, test) in enumerate(fold_dataset):
            test_folder_path = path_joiner(save_folder, str(i))
            check_dir(test_folder_path, replace=True)

            test_file_path = path_joiner(test_folder_path, f"test{ext}")
            self.write_tabular(test, test_file_path, **kwargs)

            for j, (train, val) in enumerate(train_val):
                if val is None:
                    train_file_path = path_joiner(test_folder_path, f"train{ext}")
                    self.write_tabular(train, train_file_path, **kwargs)
                    break

                val_folder_path = path_joiner(test_folder_path, str(j))
                check_dir(val_folder_path, replace=True)

                val_file_path = path_joiner(val_folder_path, f"val{ext}")
                train_file_path = path_joiner(val_folder_path, f"train{ext}")

                self.write_tabular(val, val_file_path, **kwargs)
                self.write_tabular(train, train_file_path, **kwargs)

    def write_negatives(
        self,
        neg_dict: Dict[str, List[str]],
        save_folder: str,
        sep: str = "\t",
        ext: str = ".tsv",
        scope: str = "test",
        **kwargs: Any
    ):
        check_dir(save_folder)
        file_path = path_joiner(save_folder, f"{scope}_negative{ext}")

        with open(file_path, "w") as f:
            for user_id, neg_list in neg_dict.items():
                f.write(f"{(user_id,)}{sep}" + sep.join(map(str, neg_list)) + "\n")

    def write_model(
        self,
        obj: object,
        save_folder: str,
        model_name: str
    ):
        check_dir(save_folder)
        file_path = path_joiner(save_folder, model_name, f"best-weights-{model_name}.pth")
        torch.save(obj, file_path)

        self.logger.info(
            "Model saved to disk",
            extra={"context": {"path": file_path}}
        )

    def write_recommendation(
        self,
        recommendations: dict,
        save_folder: str = "",
        model_name: str = "",
        it: Optional[int] = None,
        sep: str = "\t",
        ext: str = ".tsv"
    ):
        check_dir(save_folder)

        suffix = f"_it={it}" if it is not None else ""
        file_name = f"{model_name}{suffix}"

        file_path = path_joiner(save_folder, f"{file_name}{ext}")

        with open(file_path, 'w') as out:
            for u, recs in recommendations.items():
                for i, value in recs:
                    out.write(str(u) + sep + str(i) + sep + str(value) + '\n')


def save_np(npy, filename):
    """
    Store numpy to memory.
    Args:
        npy: numpy to save
        filename (str): filename
    """
    np.save(filename, npy)


def store_recommendation(
    recommendations: dict,
    save_folder: str = "",
    model_name: str = "",
    it: Optional[int] = None,
    sep: str = "\t",
    ext: str = ".tsv"
):
    """
    Store recommendation list (top-k)
    :return:
    """
    check_dir(save_folder)

    suffix = f"_it={it}" if it is not None else ""
    file_name = f"{model_name}{suffix}"

    path = path_joiner(save_folder, f"{file_name}{ext}")

    with open(path, 'w') as out:
        for u, recs in recommendations.items():
            for i, value in recs:
                out.write(str(u) + sep + str(i) + sep + str(value) + '\n')
