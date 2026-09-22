from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from scipy.sparse import csr_matrix
from collections import defaultdict
from functools import cached_property
from tqdm import tqdm

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


class Interactions:
    """Unordered user-item view of one split (train or eval), parallel to `Sessions`
    (which is ordered). Builds the ratings dict/sparse matrix once and exposes
    per-fold sampler dataloaders plus this split's own private view of any side
    information.

    A duplicate (user, item) pair (the same interaction repeated at different
    timestamps) is collapsed ONLY in `get_dict()`/`get_positive_items()` - a plain
    `dict` can hold just one value per key, so `_build_dict()` keeps the rating of
    the most recent occurrence by timestamp (or, absent a `timestamp` column, the
    dataframe's own last row for that pair). Every other view - `transactions`,
    `sparse_ratings`, `sparse_counts`, `sparse`, `sparse_tensor` - is built directly
    from the raw dataframe and never collapses repeats.

    Args:
        dataframe (pd.DataFrame): Interactions for this split, with at least
            'userId', 'itemId', and 'rating' columns.
        name (str): Name of this split (e.g. "train", "test", "validation").
        mappings (Tuple[Dict[Any, int], Dict[Any, int]]): (user, item) mappings from
            public ids to private indices.
        inv_mappings (Tuple[List[Any], List[Any]]): Inverse (user, item) mappings from
            private indices back to public ids.
        side_info (SideInformation, optional): Shared, centrally-cached side-information
            handle. Defaults to None.
    """

    dataframe: pd.DataFrame
    dims: Tuple[int, int]
    transactions: int
    sparse: csr_matrix
    sparse_ratings: csr_matrix
    sparse_counts: csr_matrix
    sparse_tensor: torch.Tensor

    def __init__(
        self,
        dataframe: pd.DataFrame,
        name: str,
        mappings: Tuple[Dict[Any, int], Dict[Any, int]],
        inv_mappings: Tuple[List[Any], List[Any]],
        side_info: Optional[SideInformation] = None
    ):
        # Initializing variables
        self.name = name

        self._dataframe = dataframe
        self._side_info = side_info

        # Set mappings
        self._u_map, self._i_map = mappings
        self._u_map_inv, self._i_map_inv = inv_mappings

        self._cold_items = set()
        self._cold_users = set()
        self._cached_datasets = {}

        # Build the ratings dict, in both its public-id (`_dict`) and private-id
        # (`_p_dict`) views (most callers only ever need the latter)
        self._dict, self._transactions = self._build_dict()
        self._p_dict = self._remap_ids(self._dict)

        self._users, self._items = self._get_users_and_items()
        self._n_users = len(self._users)
        self._n_items = len(self._items)

        if name == "train":
            # Eagerly materialize (and cache) the sparse ratings matrix: `Sessions`
            # needs it right away, so build it now instead of on first access
            _ = self.sparse_ratings

        # Shared by reference -- never copied/mutated. The materialized payload itself
        # lives on `self._side_info` (cached once for the whole experiment); this
        # instance's own `_side_info_cache` just holds a reference to it
        self._side_info_cache = (
            {loader_name: None for loader_name in self._side_info} if self._side_info else {}
        )

    def _build_dict(
        self,
        skip_cold_users_items: bool = True
    ) -> Tuple[Dict[Any, Dict[Any, float]], int]:
        """Convert the raw interactions DataFrame into a `dict[user -> dict[item ->
        rating]]`, keyed by public ids, tracking cold users/items encountered along
        the way.

        Args:
            skip_cold_users_items (bool): If True, skip (and record into
                `self._cold_users`/`self._cold_items`) users/items not present in
                `self._u_map`/`self._i_map`. Defaults to True.

        Returns:
            Tuple[Dict[Any, Dict[Any, float]], int]: The ratings dict (keyed by
                public ids) and the total number of (non-cold) raw rows.
        """
        ratings_dict = defaultdict(dict)
        transactions = 0

        data = self._dataframe
        if "timestamp" in data.columns:
            # Stable sort: among same-timestamp duplicates (or when every
            # timestamp is a tie, e.g. absent entirely), the dataframe's own row
            # order still breaks the tie, exactly matching the no-timestamp case
            data = data.sort_values("timestamp", kind="stable")
        users, items, ratings = data["userId"], data["itemId"], data["rating"]

        iter_df = tqdm(
            zip(users, items, ratings),
            total=len(users),
            desc=f"Building ratings dict for {self.name}",
            leave=False
        )

        for user, item, rating in iter_df:
            if skip_cold_users_items:
                # Cold user?
                if user not in self._u_map:
                    self._cold_users.add(user)
                    # And cold item?
                    if item not in self._i_map:
                        self._cold_items.add(item)
                    continue

                ratings_dict.setdefault(user, {})

                # Cold item?
                if item not in self._i_map:
                    self._cold_items.add(item)
                    continue

            # Register rating, if not cold; later rows (see the timestamp sort
            # above) win over earlier ones for the same (user, item) pair
            ratings_dict[user][item] = rating
            transactions += 1

        return dict(ratings_dict), transactions

    def _remap_ids(self, nested: Dict[Any, Dict[Any, Any]]) -> Dict[int, Dict[int, Any]]:
        """Translate a public-id-keyed nested dict (`user -> item -> value`) into
        its private-id-keyed equivalent, using `self._u_map`/`self._i_map`.

        Args:
            nested (Dict[Any, Dict[Any, Any]]): The public-id-keyed nested dict.

        Returns:
            Dict[int, Dict[int, Any]]: The same dict, keyed by private ids.
        """
        private_dict = {}

        # Translate both the outer (user) and inner (item) keys to private ids
        for user, inner in nested.items():
            mapped_user = self._u_map.get(user)
            private_dict[mapped_user] = {
                self._i_map.get(i): v for i, v in inner.items()
            }

        return private_dict

    def _get_users_and_items(self, private: bool = False) -> Tuple[List[Any], List[Any]]:
        """Collect the sorted list of distinct users and items appearing in the
        ratings dict.

        Args:
            private (bool): If True, read from `self._p_dict` (private ids); otherwise
                from `self._dict` (public ids). Defaults to False.

        Returns:
            Tuple[List[Any], List[Any]]: The (users, items) found in the dict.
        """
        ratings_dict = self._dict if not private else self._p_dict

        users = list(ratings_dict.keys())
        item_set = set()

        # Union every user's rated items into a single, deduplicated item set
        for user_ratings in ratings_dict.values():
            item_set.update(user_ratings.keys())

        items = sorted(list(item_set))

        return users, items

    def get_dataloader(
        self,
        sampler_name: str,
        batch_size: int = 1024,
        seed: int = 42,
        **kwargs: Any
    ) -> DataLoader:
        """Build (or reuse a cached) dataloader over this split's interactions, for
        the given sampler.

        Args:
            sampler_name (str): Name of the sampler registered in `sampler_registry`.
            batch_size (int): Batch size for the returned dataloader. Defaults to 1024.
            seed (int): Random seed forwarded to the sampler. Defaults to 42.
            **kwargs (Any): Additional keyword arguments forwarded to the sampler.
                An explicit `transactions` overrides the sampler's event count and,
                combined with a cache-size mismatch, forces a rebuild.

        Returns:
            DataLoader: The (possibly cached) dataloader for this sampler.
        """
        if kwargs.get('transactions') is not None:
            transactions = kwargs.pop('transactions')
        else:
            # Default event budget: one event per distinct (user, item) pair
            transactions = self.get_unique_pairs()

        # Rebuild if never cached, or if the cached dataset's own size no longer
        # matches the requested number of transactions (e.g. a different sampling
        # ratio was requested for the same sampler name)
        if (sampler_name not in self._cached_datasets or
            len(self._cached_datasets[sampler_name]) != transactions):

            users, items = self._get_users_and_items(private=True)

            sampler = sampler_registry.get(
                name=sampler_name,
                train_dict=self.get_dict(private=True),
                transactions=transactions,
                users=users,
                items=items,
                n_users=self._n_users,
                n_items=self._n_items,
                seed=seed,
                **kwargs
            )

            dataset = build_dataset(sampler)

            self._cached_datasets[sampler_name] = dataset

        dataset = self._cached_datasets[sampler_name]
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=getattr(dataset, 'collate_fn', None),
            shuffle=True
        )

        return dataloader

    def get_side_info_dataloader(
        self,
        sampler_name: str,
        batch_size: int = 1024,
        seed: int = 42,
        **kwargs: Any
    ) -> DataLoader:
        """Build (or reuse a cached) dataloader for a side-info sampler (see
        `PipelineSideInfoSampler`/`TraditionalSideInfoSampler`), entirely
        independent of this split's interaction data.

        Unlike `get_dataloader`, no `train_dict`/`users`/`items`/`n_users`/`n_items`
        are forwarded to the sampler - a side-info sampler draws from its own source
        (e.g. a knowledge graph), so it only needs `seed` plus whatever `**kwargs`
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

    def build_items_neighbour(self) -> Dict[int, List[int]]:
        """Build, for every item, the list of user (private) indices that
        interacted with it, derived from `self.sparse`.

        Returns:
            Dict[int, List[int]]: Mapping from private item index to the list of
                private user indices that interacted with it.
        """
        # (row, col) = (user, item) index pairs for every non-zero entry
        row, col = self.sparse.nonzero()
        edge_index = np.array([row, col])

        # For each item, find its edges (columns matching that item) and collect
        # the owning users (row) on the other end
        iu_dict = {
            i: edge_index[0, iu].tolist()
            for i, iu in enumerate(list((edge_index[1] == i).nonzero()[0] for i in self._items))
        }

        return iu_dict

    @cached_property
    def sparse_ratings(self) -> csr_matrix:
        """Sparse (users x items) matrix of raw ratings, built directly from the
        DataFrame: a duplicate (user, item) pair is never collapsed - repeated rows
        simply sum into the same cell (scipy's own COO->CSR construction)."""
        return self._rows_to_sparse(self._dataframe["rating"])

    @cached_property
    def sparse_counts(self) -> csr_matrix:
        """Sparse (users x items) matrix of per-pair interaction repetition counts
        (>=1 for every seen pair; >1 wherever a user repeated the same interaction).
        Suitable as an implicit-feedback confidence signal."""
        return self._rows_to_sparse(pd.Series(1.0, index=self._dataframe.index))

    @cached_property
    def sparse(self) -> csr_matrix:
        """Sparse (users x items) binary interaction matrix."""
        return self.sparse_ratings.astype(bool).astype('float32')

    @cached_property
    def sparse_tensor(self) -> torch.Tensor:
        """Sparse (users x items) COO `torch.Tensor` view of `self.sparse`."""
        coo = self.sparse.tocoo()
        indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
        values = torch.tensor(coo.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, size=coo.shape).coalesce()

    @property
    def dataframe(self) -> pd.DataFrame:
        """The raw interactions DataFrame for this split."""
        return self._dataframe

    @property
    def dims(self) -> Tuple[int, int]:
        """Number of (users, items) in this split."""
        return self._n_users, self._n_items

    @property
    def transactions(self) -> int:
        """Total number of (non-cold) raw interaction rows in this split, duplicate
        (user, item) pairs counted individually."""
        return self._transactions

    @property
    def side_information(self) -> Dict[str, Any]:
        """This instance's own cache of fold-private side-information views,
        keyed by loader name (see `get_side_info()`)."""
        return self._side_info_cache

    def get_users_items(self) -> Tuple[List[Any], List[Any]]:
        """Return the public users and items appearing in this split.

        Returns:
            Tuple[List[Any], List[Any]]: The (users, items) found in the dict.
        """
        return self._users, self._items

    def get_dict(self, private: bool = False) -> Dict[Any, Dict[Any, float]]:
        """Return the ratings dict for this split.

        Args:
            private (bool): If True, return the private-id-keyed dict; otherwise the
                public-id-keyed one. Defaults to False.

        Returns:
            Dict[Any, Dict[Any, float]]: The ratings dict.
        """
        return self._dict if not private else self._p_dict

    def get_unique_pairs(self) -> int:
        """Number of distinct (user, item) pairs in this split's ratings dict.

        Returns:
            int: The number of distinct (user, item) pairs.
        """
        return sum(len(items) for items in self._p_dict.values())

    def get_positive_items(self) -> List[List[int]]:
        """Return the list of positive (private) item indices per (private) user
        index, sorted by user index.

        Returns:
            List[List[int]]: Positive item indices per user.
        """
        users = sorted(list(self._p_dict.keys()))

        pos = []
        for u in users:
            # Deduplicate, then materialize as a plain list per user
            items_train = self._p_dict.get(u, ())
            train_set = list(set(items_train))
            pos.append(list(train_set))

        return pos

    def _rows_to_sparse(self, values: pd.Series) -> csr_matrix:
        """Build a sparse (users x items) matrix directly from this split's raw
        DataFrame, one contribution per interaction row (cold rows excluded).
        Duplicate (user, item) rows are never collapsed to a single entry.

        Args:
            values (pd.Series): Per-row values to place at each (user, item)
                coordinate (e.g. `self._dataframe["rating"]`, or a constant `1.0`
                series for occurrence counts), aligned by index with
                `self._dataframe`.

        Returns:
            csr_matrix: The sparse matrix, shaped `(len(u_map), len(i_map))`.
        """
        users = self._dataframe["userId"].map(self._u_map)
        items = self._dataframe["itemId"].map(self._i_map)
        # `.map()` on a dict yields NaN for keys absent from it: exactly the
        # (non-)cold rows this split's mappings were built to recognize
        not_cold = users.notna() & items.notna()

        rows = users[not_cold].to_numpy(dtype=np.int64)
        cols = items[not_cold].to_numpy(dtype=np.int64)
        data = values[not_cold].to_numpy(dtype=np.float64)

        return csr_matrix(
            (data, (rows, cols)), dtype=float, shape=(len(self._u_map), len(self._i_map))
        )

    def get_loader(self, name: str) -> Any:
        """Return the raw `AbstractLoader` instance registered under `name` on the
        shared `SideInformation`, e.g. to inspect `get_mapped()`/`alignment` directly,
        as opposed to `get_side_info()`, which returns its *materialized, per-fold-remapped*
        payload.

        Args:
            name (str): Name of the registered loader.

        Returns:
            Any: The `AbstractLoader` instance registered under `name`.
        """
        return self._side_info[name]

    def get_side_info(self, name: str) -> Dict[str, Any]:
        """Return this loader's payload, remapped into *this fold's* private-id view
        (see `_to_private_view()`/`elliot.dataset.modular_loaders.adapters.
        remap_embedding_payload`) and cached here so a second request skips even that.

        Args:
            name (str): Name of the registered loader.

        Returns:
            Dict[str, Any]: The loader's payload(s), keyed by payload name and
                remapped into this fold's private-id view.
        """
        if self._side_info_cache.get(name) is None:
            # Materialize (or reuse the already-cached) shared payload, then remap
            # every one of its named entries into this fold's own private-id view
            payloads = self._side_info.get_payload(name)

            self._side_info_cache[name] = {
                key: self._to_private_view(name, key, payload) for key, payload in payloads.items()
            }
            self._side_info.register_private_view(name, self)

        return self._side_info_cache[name]

    def forget_side_info(self, name: str) -> None:
        """Drop this instance's own cached private view for `name`.

        Args:
            name (str): Name of the registered loader.
        """
        self._side_info_cache.pop(name, None)

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
