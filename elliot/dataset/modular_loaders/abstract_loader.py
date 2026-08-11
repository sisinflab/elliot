from typing import Any, Dict, Optional, Set, Tuple
from logging import LoggerAdapter
from abc import ABC, abstractmethod

from elliot.namespace import SideInformationConfig
from elliot.dataset.modular_loaders.formats import Payload
from elliot.utils import logging as elog
from elliot.utils.enums import AlignmentMode, EntityAxis, Materialization
from elliot.utils.read import Reader


class AbstractLoader(ABC):
    provides: str  # e.g., "item_features", "user_features", "kg_edges"
    entity_axis: Dict[str, EntityAxis] = {}
    format: str  # e.g., "sparse", "dense", "graph"
    dims: Optional[int] = None
    alignment: AlignmentMode = AlignmentMode.DROP
    materialization: Materialization = Materialization.MEMORY
    requires_alignment: bool = True
    notes: Dict[str, str] = {}

    def __init__(
        self,
        users: Set[Any],
        items: Set[Any],
        ns: SideInformationConfig,
        logger: Optional[LoggerAdapter] = None
    ):
        self.logger = logger or elog.get_logger(self.__class__.__name__)
        self.reader = Reader(self.logger)

        # Initializing variables
        self.users = users
        self.items = items

        self._reader_config = ns.reader

        self.set_params(ns)

    @property
    def name(self) -> str:
        """This loader's registered name, i.e. its class name."""
        return self.__class__.__name__

    def set_params(self, params: SideInformationConfig):
        """Copy every field of `params` that matches one of this class' declared
        annotations (e.g. `attribute_file`, `mapping`, ...) onto `self`.

        Args:
            params (SideInformationConfig): The side-information config for this
                loader, as parsed from the experiment config.
        """
        # Only copy fields this subclass actually declares as class annotations
        for name, val in params.model_dump().items():
            if name in self.__class__.__annotations__:
                setattr(self, name, val)

    def get_mapped(self) -> Tuple[Set[Any], Set[Any]]:
        """Report this loader's current id domain, as narrowed by `__init__()` /
        `filter()`. Override only if some other attribute (a raw triples table,
        a derived feature map, ...) is the real source of truth for the domain
        and could disagree with `self.users`/`self.items`.

        Returns:
            Tuple[Set[Any], Set[Any]]: Current user and item ids sets.
        """
        return self.users, self.items

    def filter(self, users: Set[Any], items: Set[Any]):
        """Narrow `self.users`/`self.items` down to the given sets, in place.
        Override it, calling `super().filter(users, items)` first, only when
        some other attribute (a raw triples table, a derived feature map, ...)
        also needs to be narrowed down to stay consistent with the new `self.users`/`self.items`.

        Args:
            users (Set[Any]): The narrowed users domain.
            items (Set[Any]): The narrowed items domain.
        """
        self.users = self.users & users
        self.items = self.items & items

    @abstractmethod
    def load(self) -> Dict[str, Payload]:
        """Materialize this loader's heavy payload into one (or more, named) of the
        three canonical formats, for this loader's full current `self.users`/`self.items`
        domain (the whole cross-loader-intersected universe, after `__init__()`/`filter()`).
        The only place allowed to do the genuinely expensive work (sparse-matrix/feature-index
        construction, `.npy` materialization, ...).
        """
        raise NotImplementedError()

    def unload(self):
        """Optional hook: drop any large intermediate structures this loader itself
        keeps around after `load()`'s result has been consumed elsewhere.
        """
        pass
