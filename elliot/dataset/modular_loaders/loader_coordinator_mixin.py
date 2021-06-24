import importlib
import typing as t
import pandas as pd
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class LoaderCoordinator:
    def coordinate_information(self, dataframe: t.Union[pd.DataFrame, t.List],
                               sides: t.List[SimpleNamespace]=[],
                               logger: object = None) -> t.Tuple[pd.DataFrame, SimpleNamespace]:
        if isinstance(dataframe, list):
            users = set()
            items = set()
            train, test = dataframe[0]
            users = users | set(test["userId"].unique())
            items = items | set(test["itemId"].unique())
            if not isinstance(train, list):
                users = users | set(train["userId"].unique())
                items = items | set(train["itemId"].unique())
            else:
                train, val = train[0]
                users = users | set(train["userId"].unique())
                items = items | set(train["itemId"].unique())
                users = users | set(val["userId"].unique())
                items = items | set(val["itemId"].unique())
        else:
            users = set(dataframe["userId"].unique())
            items = set(dataframe["itemId"].unique())

        ns = SimpleNamespace()

        side_info_objs = []
        users_items = []
        for side in sides:
            dataloader_class = getattr(importlib.import_module("elliot.dataset.modular_loaders.loaders"), side.dataloader)
            if issubclass(dataloader_class, AbstractLoader):
                side_obj = dataloader_class(users, items, side, logger)
                side_info_objs.append(side_obj)
                users_items.append(side_obj.get_mapped())
            else:
                raise Exception("Custom Loaders must inherit from AbstractLoader")

        while True:
            new_users = users
            new_items = items
            for us_, is_ in users_items:
                new_users = new_users & us_
                new_items = new_items & is_
            if (len(new_users) == len(users)) & (len(new_items) == len(items)):
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

        if isinstance(dataframe, list):
            new_dataframe = []
            for tr, te in dataframe:
                test = self.clean_dataframe(te, users, items)
                if isinstance(tr, list):
                    train_fold = []
                    for tr_, va in tr:
                        tr_ = self.clean_dataframe(tr_, users, items)
                        va = self.clean_dataframe(va, users, items)
                        train_fold.append((tr_, va))
                else:
                    train_fold = self.clean_dataframe(tr, users, items)
                new_dataframe.append((train_fold, test))
            dataframe = new_dataframe
            # dataframe = [([(self.clean_dataframe(tr_, users, items), self.clean_dataframe(va, users, items)) for tr_, va in tr], self.clean_dataframe(te, users, items)) for tr, te in dataframe]
        else:
            dataframe = self.clean_dataframe(dataframe, users, items)

        return dataframe, ns

    def clean_dataframe(self, dataframe, users, items):
        dataframe = dataframe[dataframe['userId'].isin(users)]
        return dataframe[dataframe['itemId'].isin(items)]