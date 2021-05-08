import importlib
import typing as t
import pandas as pd
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class LoaderCoordinator:
    def coordinate_information(self, dataset_path: str,
                               sep: str="\t",
                               header: bool=False,
                               names: t.List=[],
                               sides: t.List[SimpleNamespace]=[]) -> t.Tuple[pd.DataFrame, SimpleNamespace]:

        dataframe = pd.read_csv(dataset_path, sep=sep, header=header, names=names)
        users = set(dataframe["userId"].unique())
        items = set(dataframe["itemId"].unique())
        ns = SimpleNamespace()

        side_info_objs = []
        users_items = []
        for side in sides:
            dataloader_class = getattr(importlib.import_module("elliot.dataset.modular_loaders.loaders"), side.dataloader)
            if issubclass(dataloader_class, AbstractLoader):
                side_obj = dataloader_class(dataframe, side)
                side_info_objs.append(side_obj)
                users_items.append(side_obj.get_mapped())
            else:
                raise Exception("Custom Loaders must inherit from AbstractLoader")

        while True:
            new_users = users
            new_items = items
            for us_, is_ in users_items:
                new_users = new_users and us_
                new_items = new_items and is_
            if (len(new_users) == len(users)) and (len(new_items) == len(items)):
                break
            else:
                users = new_users
                items = new_items

                for side_obj in side_info_objs:
                    side_obj.filter(users, items)

        for side_obj in side_info_objs:
            side_ns = side_obj.create_namespace()
            name = side_ns.__name__
            setattr(ns, name, side_ns)

        dataframe = dataframe[dataframe['userId'].isin(users)]
        dataframe = dataframe[dataframe['itemId'].isin(items)]

        return dataframe, ns