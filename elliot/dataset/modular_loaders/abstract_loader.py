from typing import Tuple, Dict, Optional, Set
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
        users: Set,
        items: Set,
        ns: SideInformationConfig,
        logger: Optional[LoggerAdapter] = None
    ):
        """Pure configuration step -- no I/O. `users`/`items` are the *unfiltered* base
        universe; `discover()` (called explicitly, once, right after construction) is
        what actually narrows them down to this loader's own domain.
        """
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

    def get_mapped(self) -> Tuple[Set[int], Set[int]]:
        """Report this loader's current id domain, as narrowed by `discover()`/
        `filter()`. The default -- just `self.users`/`self.items` -- is correct for
        every loader that keeps no other id-keyed state; override only if some other
        attribute (a raw triples table, a derived feature map, ...) is the real source
        of truth for the domain and could disagree with `self.users`/`self.items`.
        """
        return self.users, self.items

    def filter(self, users: Set[int], items: Set[int]) -> None:
        """Narrow `self.users`/`self.items` down to the given sets, in place. Called
        exactly once, globally, by `DataSetLoader._intersect_users_items()` -- before
        any `DataSet`/fold exists and before this loader is ever shared across owners.
        Never call this per fold: with a single `SideInformation` shared by reference
        across every fold, mutating a loader here would corrupt it for every other
        fold/owner still holding it. Per-fold scoping instead happens non-destructively
        in `load()`.

        The default just intersects `self.users`/`self.items`. Override it -- calling
        `super().filter(users, items)` first -- only when some other attribute set up in
        `discover()` (a raw triples table, a derived feature map, ...) also needs to be
        narrowed down to stay consistent with the new `self.users`/`self.items`.
        """
        self.users = self.users & users
        self.items = self.items & items

    @abstractmethod
    def load(self) -> Dict[str, Payload]:
        """Materialize this loader's heavy payload into one (or more, named) of the
        three canonical formats (`EmbeddingPayload`/`TextPayload`/`GraphPayload`, see
        `elliot.dataset.modular_loaders.formats`), for this loader's full current
        `self.users`/`self.items` domain (the whole cross-loader-intersected universe,
        after `discover()`/`filter()`). The only place allowed to do the genuinely
        expensive work (sparse-matrix/feature-index construction, `.npy` materialization,
        ...).

        Called at most once for the whole experiment: `SideInformation.get_payload()`
        caches the result and hands the identical object to every fold/owner that asks
        for it. Callers that need a subset (e.g. one batch's users/items) slice the
        returned payload themselves via `elliot.dataset.modular_loaders.adapters`
        (`embedding_to_dense(payload, ids=...)`, ...) -- this method never produces a
        per-caller copy.
        """
        raise NotImplementedError()

    def unload(self) -> None:
        """Optional hook: drop any large intermediate structures this loader itself
        keeps around after `load()`'s result has been consumed elsewhere. Default is a
        no-op. Called by `SideInformation.marked_as_done()` once every model that
        declared this loader has finished every one of its folds -- unlike
        `forget_side_info()` (which only drops one owner's own reference to the shared
        payload), this is the point where it's actually safe to release the payload for
        good, since nothing else in the experiment still needs it.
        """
        pass
