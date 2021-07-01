import copy
import sys
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

    if float(".".join([str(sys.version_info[0]), str(sys.version_info[1])])) < 3.8:
        _version_warning = "WARNING: Your Python version is lower than 3.8. Consequently, Custom class objects created in Side Information Namespace will be created swallowly!!!!"
        print(_version_warning, file=sys.stderr)

        def __deepcopy__(self, memo = {}):
            self.logger.warning(self._version_warning)
            newself = object.__new__(self.__class__)
            for method_name in dir(self.__class__):
                newself.__dict__[method_name] = getattr(self, method_name)
            for attribute_name, attribute_value in self.__dict__.items():
                if attribute_value.__class__.__module__ == "builtins":
                    newself.__dict__[attribute_name] = copy.deepcopy(attribute_value)
                else:
                    newself.__dict__[attribute_name] = attribute_value
            return newself

