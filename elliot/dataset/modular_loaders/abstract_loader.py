import copy
from abc import ABC, abstractmethod
import typing as t
from types import SimpleNamespace


class AbstractLoader(ABC):
    @abstractmethod
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace):
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

    def __deepcopy__(self, memo = {}):
        newself = object.__new__(self.__class__)
        for method_name in dir(self.__class__):
            newself.__dict__[method_name] = getattr(self, method_name)
        for attribute_name, attribute_value in self.__dict__.items():
            if attribute_value.__class__.__module__ == "builtins":
                newself.__dict__[attribute_name] = copy.deepcopy(attribute_value)
            else:
                newself.__dict__[attribute_name] = attribute_value
        return newself
