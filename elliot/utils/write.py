"""
Module description:

"""

import json
from typing import List, Tuple, Dict, Any, Callable, Optional, Union
from logging import LoggerAdapter
import torch
import pandas as pd
from datetime import datetime

from elliot.utils.folder import check_dir, path_joiner
from elliot.utils.logging import get_logger


def _timestamp():
    return datetime.now().strftime("%Y_%m_%d_%H_%M_%S")


class Writer:
    """Utility class for writing various types of data files.

    Attributes:
        logger (LoggerAdapter): A logging instance.
    """

    def __init__(self, logger: LoggerAdapter = get_logger("__main__")):
        self.logger = logger

    def write_tabular(
        self,
        data: pd.DataFrame,
        path: str,
        header: Union[bool, List[str]] = False,
        columns: Optional[List[Union[str, int]]] = None,
        sep: str = "\t",
        callback_fn: Optional[Callable] = None,
        **kwargs: Any
    ):
        """Write a DataFrame to a file in tabular format.

        Args:
            data (pd.DataFrame): DataFrame to write to the file.
            path (str): Path to the output file.
            header (Union[bool, List[str]]): Whether to write a header row in the output file. Defaults to False.
                If a list of strings is given, it is assumed to be aliases for the column names.
            columns (List[Union[str, int]], optional): List of column names or indices
                to select. Defaults to None.
            sep (str): Column separator to use in the output file. Defaults to "\\t".
            callback_fn (Callable, optional): Function to call after writing the file. Defaults to None.
            **kwargs (Any): Additional keyword arguments passed to the `callback_fn` function.
        """
        # Check whether columns are specified as positional indices
        is_positional = columns is not None and any(isinstance(c, int) for c in columns)

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
                    "None of the desired column indices were found. Saving an empty DataFrame."
                )
                df = pd.DataFrame()
            else:
                df = data.iloc[:, valid_idx]

        # Case 3: semantic column selection by name
        else:
            cols_to_use = [c for c in columns if c in data.columns]
            if not cols_to_use:
                self.logger.warning(
                    "None of the desired columns were found. Saving an empty DataFrame."
                )
                df = pd.DataFrame()
            else:
                df = data[cols_to_use]

        # Check whether the header should be written
        if isinstance(header, list) and len(header) != len(df.columns):
            self.logger.warning(
                "`header` length does not match `data` selected columns count. Saving with no header."
            )
            header = False

        df.to_csv(path, sep=sep, index=False, header=header)

        self.logger.info(f"Saved: {path}")

        if callback_fn is not None:
            callback_fn(df, **kwargs)

    def write_tabular_split(
        self,
        fold_dataset: List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]],
        save_folder: str,
        ext: str = ".tsv",
        **kwargs: Any
    ):
        """Write tabular dataset splits in a structured manner.

        Args:
            fold_dataset (List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], pd.DataFrame]]):
                List of tuples containing a list of (train, val) dataset tuples and a test DataFrame.
            save_folder (str): Path to the folder where the datasets will be saved.
            ext (str): File extension for the output files. Defaults to ".tsv".
            **kwargs: Additional keyword arguments passed to the `write_tabular` method.
        """
        check_dir(save_folder, replace=True)

        # Test fold level
        for i, (train_val, test) in enumerate(fold_dataset):
            test_folder_path = path_joiner(save_folder, str(i))
            check_dir(test_folder_path, replace=True)

            test_file_path = path_joiner(test_folder_path, f"test{ext}")
            self.write_tabular(data=test, path=test_file_path, **kwargs)

            # Validation fold level
            for j, (train, val) in enumerate(train_val):

                # Save only train dataset if val dataset is missing
                if val is None:
                    train_file_path = path_joiner(test_folder_path, f"train{ext}")
                    self.write_tabular(data=train, path=train_file_path, **kwargs)
                    break

                val_folder_path = path_joiner(test_folder_path, str(j))
                check_dir(val_folder_path, replace=True)

                val_file_path = path_joiner(val_folder_path, f"val{ext}")
                train_file_path = path_joiner(val_folder_path, f"train{ext}")

                self.write_tabular(data=val, path=val_file_path, **kwargs)
                self.write_tabular(data=train, path=train_file_path, **kwargs)

    def write_negatives(
        self,
        neg_dict: Dict[str, List[str]],
        save_folder: str,
        sep: str = "\t",
        ext: str = ".tsv",
        scope: str = "test",
        **kwargs: Any
    ):
        """Write negative samples from a dictionary to a delimited file.

        Args:
            neg_dict (Dict[str, List[str]]): Dictionary containing user IDs as keys and lists of
                negative samples as values.
            save_folder (str): Path to the folder where the output file will be saved.
            sep (str, optional): Field separator to use in the output file. Defaults to "\\t".
            ext (str, optional): File extension for the output file. Defaults to ".tsv".
            scope (str, optional): File naming scope to include in the filename. Defaults to "test".
            **kwargs (Any): Additional keyword arguments.
        """
        check_dir(save_folder)
        path = path_joiner(save_folder, f"{scope}_negative{ext}")

        with open(path, "w") as f:
            for user_id, neg_list in neg_dict.items():
                f.write(f"{(user_id,)}{sep}" + sep.join(map(str, neg_list)) + "\n")

        self.logger.info(f"Saved: {path}")

    def write_model(
        self,
        obj: object,
        save_folder: str,
        model_name: str,
        ext: str = ".pth",
        **kwargs: Any
    ):
        """Save the model object to the specified folder with the given name.

        Args:
            obj (object): Model object to be saved.
            save_folder (str): Path to the folder where the model will be saved.
            model_name (str): Name of the model, used to generate the file name.
            ext (str, optional): File extension for the output file. Defaults to ".pth".
            **kwargs (Any): Additional keyword arguments.
        """
        check_dir(save_folder)
        file_path = path_joiner(save_folder, model_name, f"best-weights-{model_name}{ext}")
        torch.save(obj, file_path)

        self.logger.info(
            "Model saved to disk",
            extra={"context": {"path": file_path}}
        )

    def write_recommendations(
        self,
        recommendations: dict,
        save_folder: str,
        model_name: str,
        it: Optional[int] = None,
        ext: str = ".tsv",
        **kwargs: Any
    ):
        """Write top-k recommendations to a file with specified formatting.

        Args:
            recommendations (dict): Dictionary where keys are user IDs and values
                are lists of tuples containing the top-k item IDs and their associated scores.
            save_folder (str): Path to the folder where the file will be saved.
            model_name (str): Name of the model to use as part of the file name.
            it (int, optional): Iteration number to include in the file name. Defaults to None.
            ext (str): File extension for the output files. Defaults to ".tsv".
            **kwargs (Any): Additional keyword arguments passed to the `write_tabular` method.
        """
        check_dir(save_folder)

        suffix = f"_it={it}" if it is not None else ""
        file_name = f"{model_name}{suffix}"

        path = path_joiner(save_folder, f"{file_name}{ext}")

        rows = [
            (u, i, value)
            for u, recs in recommendations.items()
            for i, value in recs
        ]
        df = pd.DataFrame(rows)

        self.write_tabular(data=df, path=path, **kwargs)

    def write_results(
        self,
        results: Dict[int, Dict[str, Any]],
        save_folder: str,
        file_name: str = "",
        ext: str = ".tsv",
        triplets: bool = False,
        **kwargs: Any
    ):
        """Write results data to files with specified formatting.

        Args:
            results (Dict[int, Dict[str, Any]]): Dictionary where keys are integer cutoffs
                and values are nested dictionaries containing the data to save.
            save_folder (str): Path to the folder where the files will be saved.
            file_name (str): Optional base name for the output files. Defaults to "".
            ext (str): File extension for the output files. Defaults to ".tsv".
            triplets (bool): If True, outputs the data in tabular format (as triplets). Defaults to False.
            **kwargs (Any): Additional keyword arguments passed to the writing methods.
        """
        check_dir(save_folder)

        for k, data in results.items():
            if not data:
                continue

            prefix = "triplets_rec" if triplets else "rec"
            name = f"{prefix}_cutoff_{k}{file_name}_{_timestamp()}"

            if triplets:
                info = (
                    pd.DataFrame.from_dict(data, orient="index")
                    .stack()
                    .reset_index()
                )
                info.columns = ["model", "metric", "value"]

                self.write_tabular(
                    data=info,
                    path=path_joiner(save_folder, f"{name}{ext}"),
                    **kwargs
                )
            else:
                self.write_dict_as_table(
                    data=data,
                    save_folder=save_folder,
                    file_name=name,
                    ext=ext,
                    **kwargs
                )

    def write_times(
        self,
        data: Dict[str, Dict[str, Any]],
        save_folder: str,
        file_name: str = "",
        **kwargs: Any
    ):
        """Write timing data to a file.

        Args:
            data (Dict[str, Dict[str, Any]]): Dictionary containing timing data.
            save_folder (str): Path to the folder where the file will be saved.
            file_name (str): Optional base name for the output file. Defaults to "".
            **kwargs (Any): Additional keyword arguments passed to the `write_dict_as_table` method.
        """
        name = f"rec_training_time{file_name}_{_timestamp()}"

        self.write_dict_as_table(
            data=data,
            save_folder=save_folder,
            file_name=name,
            **kwargs
        )

    def write_trials(
        self,
        trials: Dict[str, Any],
        save_folder: str,
        file_name: str = "",
        frmt: str = "json",
        ext: str = ".tsv",
        **kwargs: Any
    ):
        """Write trials data to specified file format.

        Args:
            trials (Dict[str, Any]): Dictionary where keys are model names and values
                are lists of trials data.
            save_folder (str): Path to the folder where the files will be saved.
            file_name (str): Optional base name for the output files. Defaults to "".
            frmt (str): File format to save the trials' data. Defaults to "json".
                Supported values are "json" and "tabular".
            ext (str): File extension for the tabular output files. Defaults to ".tsv".
            **kwargs (Any): Additional keyword arguments passed to the writing methods.
        """
        check_dir(save_folder)

        for model_name, trials_list in trials.items():
            if not trials:
                continue

            name = f"trials_{model_name}{file_name}_{_timestamp()}"

            if frmt == "json":
                name_ = name + ".json"
                path = path_joiner(save_folder, name_)
                self.write_json(data=trials_list, path=path, **kwargs)

            elif frmt == "tabular":
                info = pd.DataFrame(trials_list)
                name_ = name + ext
                path = path_joiner(save_folder, name_)
                self.write_tabular(data=info, path=path, **kwargs)

    def write_params(
        self,
        params: List[Dict[str, Any]],
        save_folder: str,
        file_name: str = "",
        **kwargs: Any
    ):
        """Write parameter data to a JSON file.

        Args:
            params (List[Dict[str, Any]]): List of parameter dictionaries to write to the JSON file.
            save_folder (str): Path to the folder where the JSON file will be saved.
            file_name (str): Optional base name for the output file. Defaults to "".
            **kwargs (Any): Additional keyword arguments passed to the `write_json` method.
        """
        check_dir(save_folder)

        default_k = params[0].get("default_validation_cutoff")

        name = f"bestmodelparams_cutoff_{default_k}{file_name}_{_timestamp()}.json"
        path = path_joiner(save_folder, name)

        self.write_json(data=params, path=path, indent=4, **kwargs)

    def write_statistical_results(
        self,
        results: Dict[int, Any],
        save_folder: str,
        file_name: str = "",
        ext: str = ".tsv",
        stat_test: str = "",
        **kwargs: Any
    ):
        """Write statistical results to formatted tabular files.

        Args:
            results (Dict[int, Any]): Mapping of cutoff values to statistical data.
            save_folder (str): Path to the folder where results will be saved.
            file_name (str): Optional base name for the output files. Defaults to "".
            ext (str): File extension for the output files. Defaults to ".tsv".
            stat_test (str): Name of the statistical test to include in the file naming. Defaults to "".
            **kwargs (Any): Additional keyword arguments passed to the `write_tabular` method.
        """
        if not results:
            return

        check_dir(save_folder)

        for k, data in results.items():
            info = pd.DataFrame(data)

            name = f"stat_{stat_test}_cutoff_{k}{file_name}_{_timestamp()}{ext}"

            self.write_tabular(
                data=info,
                path=path_joiner(save_folder, name),
                **kwargs
            )

    def write_dict_as_table(
        self,
        data: Dict[str, Dict[str, Any]],
        save_folder: str,
        file_name: str,
        ext: str = ".tsv",
        **kwargs: Any
    ):
        """Write a dictionary as a tabular file.

        Args:
            data (Dict[str, Dict[str, Any]]): Dictionary where keys are indices and values are dictionaries
                representing rows in the table.
            save_folder (str): Path to the folder where the file will be saved.
            file_name (str): Name of the output file. Defaults to "".
            ext (str): File extension for the output file. Defaults to ".tsv".
            **kwargs (Any): Additional keyword arguments passed to the `write_tabular` method.
        """
        if not data:
            return

        check_dir(save_folder)

        info = pd.DataFrame.from_dict(data, orient="index")
        info.insert(0, "model", info.index)

        self.write_tabular(
            data=info,
            path=path_joiner(save_folder, f"{file_name}{ext}"),
            **kwargs
        )

    def write_json(
        self,
        data: Any,
        path: str,
        indent: int = 2,
        **kwargs: Any
    ):
        """Write data to a JSON file.

        Args:
            data (Any): Data to be serialized and written to the JSON file.
            path (str): Path to the output file.
            indent (int): Number of spaces to use for indentation in the JSON output. Defaults to 2.
            **kwargs (Any): Additional keyword arguments.
        """
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=indent)

        self.logger.info(f"Saved: {path}")


# def save_np(npy, filename):
#     """
#     Store numpy to memory.
#     Args:
#         npy: numpy to save
#         filename (str): filename
#     """
#     np.save(filename, npy)


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
