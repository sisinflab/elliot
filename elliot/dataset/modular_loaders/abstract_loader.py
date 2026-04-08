from typing import Tuple, Dict, Optional, Set
from types import SimpleNamespace
from logging import LoggerAdapter
import copy
import logging
import sys
from packaging import version
from abc import ABC, abstractmethod

from elliot.namespace import SideInformationConfig
from elliot.utils import logging as elog
from elliot.utils.enums import AlignmentMode, Materialization
from elliot.utils.read import Reader


class AbstractLoader(ABC):
    provides: str  # e.g., "item_features", "user_features", "kg_edges"
    format: str  # e.g., "sparse", "dense", "graph"
    dims: Optional[int] = None
    alignment: AlignmentMode = AlignmentMode.DROP
    materialization: Materialization = Materialization.MEMORY
    requires_alignment: bool = True
    notes: Dict[str, str] = {}

    def __init__(
        self,
        users: Set,
        items: Set,
        ns: SideInformationConfig,
        logger: LoggerAdapter = None
    ):
        self.logger = logger or elog.get_logger(self.__class__.__name__)
        self.reader = Reader(self.logger)

        self.users = users
        self.items = items
        self._reader_config = ns.reader

        self.set_params(ns)

    @property
    def name(self):
        return self.__class__.__name__

    def set_params(self, params: SideInformationConfig):
        for name, val in params.model_dump().items():
            if name in self.__class__.__annotations__:
                setattr(self, name, val)

    @abstractmethod
    def get_mapped(self) -> Tuple[Set[int], Set[int]]:
        raise NotImplementedError()

    @abstractmethod
    def filter(self, users: Set[int], items: Set[int]):
        raise NotImplementedError()

    @abstractmethod
    def create_namespace(self) -> SimpleNamespace:
        raise NotImplementedError()

    # if version.parse(sys.version.split()[0]) < version.parse("3.8"):
    #     _version_warning = (
    #         "WARNING: Your Python version is lower than 3.8. Consequently, "
    #         "Custom class objects created in Side Information Namespace will be created shallowly."
    #     )
    #     logging.getLogger(__name__).warning(_version_warning)
    #
    #     def __deepcopy__(self, memo = {}):
    #         self.logger.warning(self._version_warning)
    #         newself = object.__new__(self.__class__)
    #         for method_name in dir(self.__class__):
    #             newself.__dict__[method_name] = getattr(self, method_name)
    #         for attribute_name, attribute_value in self.__dict__.items():
    #             if attribute_value.__class__.__module__ == "builtins":
    #                 newself.__dict__[attribute_name] = copy.deepcopy(attribute_value)
    #             else:
    #                 newself.__dict__[attribute_name] = attribute_value
    #         return newself
