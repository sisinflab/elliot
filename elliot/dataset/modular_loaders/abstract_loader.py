from abc import ABC, abstractmethod
import typing as t
from types import SimpleNamespace

import pandas as pd


class AbstractLoader(ABC):
    @abstractmethod
    def __init__(self, data: pd.DataFrame, ns: SimpleNamespace):
        raise NotImplementedError

    @abstractmethod
    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        raise NotImplementedError

    @abstractmethod
    def filter(self, users: t.Set[int], items: t.Set[int]):
        raise NotImplementedError

    @abstractmethod
    def create_namespace(self) -> SimpleNamespace:
        raise NotImplementedError
