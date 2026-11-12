from typing import Any, Dict, Iterator, List, Optional, Tuple
from torch.utils.data import DataLoader

from elliot.dataset.modular_loaders.cache import SideInformation
from elliot.dataset.modular_loaders.remap import (
    remap_embedding_payload,
    remap_graph_payload,
    remap_pair_payload,
    remap_text_payload,
)
from elliot.dataset.modular_loaders.formats import EmbeddingPayload, GraphPayload, TextPayload
from elliot.dataset.samplers.base_data import build_dataset
from elliot.utils.enums import EntityAxis
from elliot.utils.registry import sampler_registry


class FoldSideInfo:
    """One fold's private view of the experiment's side information. Built once per
    fold by `DataSet` and handed to the recommender alongside the fold's train
    `Interactions`/`Sessions`.

    The materialized payloads themselves live on the shared `SideInformation` (cached
    once for the whole experiment): this object only remaps them into the fold's
    private-id view (see `get()`) and caches that view, plus the datasets of any
    side-info sampler (see `get_dataloader()`).

    Args:
        side_info (SideInformation): Shared, centrally-cached side-information handle.
        mappings (Tuple[Dict[Any, int], Dict[Any, int]]): The fold's (user, item)
            mappings from public ids to private indices.
        inv_mappings (Tuple[List[Any], List[Any]]): The fold's inverse (user, item)
            mappings from private indices back to public ids.
    """

    def __init__(
        self,
        side_info: SideInformation,
        mappings: Tuple[Dict[Any, int], Dict[Any, int]],
        inv_mappings: Tuple[List[Any], List[Any]]
    ):
        self._side_info = side_info

        # Set mappings
        self._u_map, self._i_map = mappings
        self._u_map_inv, self._i_map_inv = inv_mappings

        # Shared by reference - never copied/mutated. `_views` just holds this fold's
        # remapped view of each payload, built on first request
        self._views: Dict[str, Optional[Dict[str, Any]]] = {
            loader_name: None for loader_name in self._side_info
        }
        self._cached_datasets = {}

    def __contains__(self, name: object) -> bool:
        return name in self._side_info

    def __iter__(self) -> Iterator[str]:
        return iter(self._side_info)

    def __len__(self) -> int:
        return len(self._side_info)

    def get_loader(self, name: str) -> Any:
        """Return the raw `AbstractLoader` instance registered under `name` on the
        shared `SideInformation`, e.g. to inspect `get_mapped()`/`alignment` directly,
        as opposed to `get()`, which returns its *materialized, per-fold-remapped*
        payload.

        Args:
            name (str): Name of the registered loader.

        Returns:
            Any: The `AbstractLoader` instance registered under `name`.
        """
        return self._side_info[name]

    def get(self, name: str) -> Dict[str, Any]:
        """Return this loader's payload, remapped into *this fold's* private-id view
        (see `_to_private_view()`/`elliot.dataset.modular_loaders.remap`) and cached
        here so a second request skips even that.

        Args:
            name (str): Name of the registered loader.

        Returns:
            Dict[str, Any]: The loader's payload(s), keyed by payload name and
                remapped into this fold's private-id view.
        """
        if self._views.get(name) is None:
            # Materialize (or reuse the already-cached) shared payload, then remap
            # every one of its named entries into this fold's own private-id view
            payloads = self._side_info.get_payload(name)

            self._views[name] = {
                key: self._to_private_view(name, key, payload) for key, payload in payloads.items()
            }
            self._side_info.register_private_view(name, self)

        return self._views[name]

    def forget(self, name: str) -> None:
        """Drop this fold's own cached private view for `name`.

        Args:
            name (str): Name of the registered loader.
        """
        self._views.pop(name, None)

    def get_dataloader(
        self,
        sampler_name: str,
        batch_size: int = 1024,
        seed: int = 42,
        **kwargs: Any
    ) -> DataLoader:
        """Build (or reuse a cached) dataloader for a side-info sampler (see
        `PipelineSideInfoSampler`/`TraditionalSideInfoSampler`), entirely
        independent of the fold's interaction data.

        Unlike `Interactions.get_dataloader`, no `train_dict`/`users`/`items`/`n_users`/
        `n_items` are forwarded to the sampler - a side-info sampler draws from its own
        source (e.g. a knowledge graph), so it only needs `seed` plus whatever `**kwargs`
        itself declares.

        Args:
            sampler_name (str): Name of the sampler registered in `sampler_registry`.
            batch_size (int): Batch size for the returned dataloader. Defaults to 1024.
            seed (int): Random seed forwarded to the sampler. Defaults to 42.
            **kwargs (Any): Additional keyword arguments forwarded to the sampler.

        Returns:
            DataLoader: The (possibly cached) dataloader for this sampler.
        """
        if sampler_name not in self._cached_datasets:
            sampler = sampler_registry.get(
                name=sampler_name,
                seed=seed,
                **kwargs
            )
            self._cached_datasets[sampler_name] = build_dataset(sampler)

        dataset = self._cached_datasets[sampler_name]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=getattr(dataset, 'collate_fn', None),
            shuffle=True
        )

        return dataloader

    def _to_private_view(self, name: str, key: str, payload: Any) -> Any:
        """Remap one named payload of loader `name` into this fold's private-id view,
        dispatching on the payload's declared `entity_axis` (see
        `elliot.dataset.modular_loaders.remap`).

        Args:
            name (str): Name of the registered loader.
            key (str): Name of the payload within the loader's payload dict.
            payload (Any): The shared, public-id-keyed payload to remap.

        Returns:
            Any: The payload, remapped into this fold's private-id view (or
                untouched, if its entity axis is `EntityAxis.NONE`).
        """
        axis = self._side_info[name].entity_axis.get(key, EntityAxis.NONE)

        # Not user/item-indexed (e.g. a shared vocabulary axis): nothing to remap
        if axis is EntityAxis.NONE:
            return payload

        # (user, item)-pair-indexed: both axes remap together
        if axis is EntityAxis.PAIR:
            return remap_pair_payload(payload, self._u_map, self._i_map)

        # Single-axis (user- or item-indexed): remap via the matching inverse mapping
        mapping = self._u_map_inv if axis is EntityAxis.USER else self._i_map_inv
        if isinstance(payload, EmbeddingPayload):
            return remap_embedding_payload(payload, mapping)
        if isinstance(payload, TextPayload):
            return remap_text_payload(payload, mapping)
        if isinstance(payload, GraphPayload):
            return remap_graph_payload(payload, mapping)

        return payload
