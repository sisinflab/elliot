from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import rows_to_embedding_payload
from elliot.dataset.modular_loaders.materialize import embedding_to_dense
from elliot.dataset.modular_loaders.formats import EmbeddingPayload
from elliot.utils.enums import AlignmentMode, EntityAxis, Materialization
from elliot.utils.folder import path_joiner
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="interaction_features",
    format="embedding",
    alignment=AlignmentMode.PAD,
    materialization=Materialization.LAZY,
    entity_axis={"interaction_features": EntityAxis.PAIR}
)
class InteractionsTextualAttributes(AbstractLoader):
    """One precomputed dense feature vector per *interaction* (e.g. a review-text
    embedding), read from a folder of `.npy` files keyed by a per-interaction id (the
    3rd column of `interactions`) - unlike `TextualAttribute`, this loader's row
    identity is the `(user, item)` pair.
    """

    interaction_ids: str
    interaction_features: str

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._interactions_df: pd.DataFrame = pd.DataFrame()
        self._interaction_features_shape: Tuple[int, ...] = ()

        self._interactions_df = self.reader.read_tabular(
            path=self.interaction_ids,
            header=self._reader_config.header,
            sep=self._reader_config.sep,
            encoding=self._reader_config.encoding
        )

        # Sniff the per-interaction feature shape from one sample file
        feature_files = self.reader.read_folder(
            folder=self.interaction_features,
            ext=".npy"
        )
        self._interactions_features_shape = self.reader.peek_npy_shape(feature_files[0])

        # Narrow the user/item domain to those actually referenced in an interaction
        self.users = self.users & set(self._interactions_df[0].unique().tolist())
        self.items = self.items & set(self._interactions_df[1].unique().tolist())

    def get_all_features(self) -> np.ndarray:
        """Dense `(n_interactions, dim)` matrix of interaction features, read through
        `load()` so it honors `self.materialization`.

        Returns:
            np.ndarray: The dense interaction-features matrix.
        """
        return embedding_to_dense(self.load()["interaction_features"])

    def load(self) -> Dict[str, EmbeddingPayload]:
        """Build the per-interaction `EmbeddingPayload` via the shared
        `rows_to_embedding_payload` dispatch (honors `self.materialization` there, not
        here).

        Returns:
            Dict[str, EmbeddingPayload]: The `interaction_features` payload.
        """
        # Keep only interactions whose user and item are both in the current domain
        all_interactions = self._interactions_df
        active = (all_interactions[
            all_interactions[0].isin(self.users) & all_interactions[1].isin(self.items)
        ])
        active = active.sort_values(by=[0, 1]).reset_index(drop=True)

        # Row identity is the (user, item) pair; the filename is this interaction's own id
        row_ids = list(zip(active[0].tolist(), active[1].tolist()))
        id_map = {pair: idx for idx, pair in enumerate(row_ids)}
        key_by_pair = dict(zip(row_ids, active[2].tolist()))

        shape = (
            tuple(self._interactions_features_shape)
            if self._interactions_features_shape else ()
        )

        def row_reader(
            pair,
            mmap_mode,
            _folder=self.interaction_features,
            _reader=self.reader,
            _keys=key_by_pair
        ):
            return _reader.read_npy(
                path_joiner(_folder, f"{_keys[pair]}.npy"), mmap_mode=mmap_mode
            )

        payload = rows_to_embedding_payload(self.materialization, id_map, shape, row_reader)
        return {"interaction_features": payload}
