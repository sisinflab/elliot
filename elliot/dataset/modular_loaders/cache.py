from typing import Dict, Iterable, Iterator, Optional, Type
import weakref
from collections.abc import Mapping

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.formats import Payload


class SideInformation(Mapping):
    """Shared, centrally-cached handle to the side-information loaders discovered for one
    experiment. A single instance of this class is created once by `DataSetLoader` and
    held, by reference, on every fold's `DataSet`/`Interactions`/`Sessions` - it is never
    copied. It materializes and caches each loader's payload exactly once for the whole
    experiment (`get_payload()`), and reference-counts it across models (`mapped_uses()`/
    `marked_as_done()`) so it can be dropped once nothing still needs it.
    """

    _loaders: Dict[str, AbstractLoader]
    _payloads: Dict[str, Optional[Dict[str, Payload]]]

    def __init__(self, loaders: Dict[str, AbstractLoader]):
        self._loaders = loaders
        self._payloads = {name: None for name in loaders}
        self._remaining_uses: Dict[str, int] = {name: 0 for name in loaders}
        self._private_view_owners: Dict[str, "weakref.WeakSet"] = {
            name: weakref.WeakSet() for name in loaders
        }

    def __getitem__(self, name: str) -> AbstractLoader:
        return self._loaders[name]

    def __iter__(self) -> Iterator[str]:
        return iter(self._loaders)

    def __len__(self) -> int:
        return len(self._loaders)

    def get_payload(self, name: str) -> Dict[str, Payload]:
        """Materialize (once) and cache the given loader's payload for the whole
        experiment. Every caller gets back the identical dict/array objects.

        Args:
            name (str): Name of the registered loader.

        Returns:
            Dict[str, Payload]: The loader's payload(s), keyed by payload name.
        """
        if self._payloads[name] is None:
            self._payloads[name] = self._loaders[name].load()
        return self._payloads[name]

    def register_private_view(self, name: str, owner):
        """Record that `owner` (an `Interactions` instance) has cached its own
        fold-private view derived from this loader's payload, so `marked_as_done()`
        can tell it to drop that view once nobody still needs the loader.

        Args:
            name (str): Name of the registered loader.
            owner (Interactions): The `Interactions` instance that cached a
                fold-private view derived from this loader's payload.
        """
        owners = self._private_view_owners.get(name)
        if owners is not None:
            owners.add(owner)

    def mapped_uses(self, model_classes: Iterable[Type]):
        """Precompute, for every loader, how many of the given model classes declare
        it via their `_loaders` class attribute. Call once, before any model runs
        (typically with every model configured for this experiment).

        Args:
            model_classes (Iterable[Type]): The model classes configured for this
                experiment.
        """
        # Tally how many configured models declare each loader
        counts = {name: 0 for name in self._loaders}
        for model_cls in model_classes:
            for loader_name in getattr(model_cls, "_loaders", []):
                if loader_name in counts:
                    counts[loader_name] += 1
        self._remaining_uses = counts

    def marked_as_done(self, name: str):
        """Report that one model declaring this loader has finished every one of its
        folds. Once every model that `mapped_uses()` counted for this loader has
        reported done: its cached shared payload is dropped, every fold's own cached
        private view derived from it is dropped too (see `register_private_view()`),
        and `unload()` is called on the loader - so all of it becomes eligible for
        garbage collection, not just this object's own reference.

        Args:
            name (str): Name of the registered loader.
        """
        # Never counted (mapped_uses() didn't see it): nothing to release
        if name not in self._remaining_uses:
            return

        self._remaining_uses[name] = max(0, self._remaining_uses[name] - 1)

        # Once every model that needed this loader has reported done, release everything
        if self._remaining_uses[name] == 0 and self._payloads[name] is not None:
            self._payloads[name] = None
            for owner in list(self._private_view_owners[name]):
                owner.forget_side_info(name)
            self._private_view_owners[name].clear()
            self._loaders[name].unload()
