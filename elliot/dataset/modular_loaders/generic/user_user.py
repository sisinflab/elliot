from typing import Any, Dict, List, Tuple

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import (
    pairwise_ids_from_raw,
    pairwise_raw_to_embedding_payload,
    public_id_map
)
from elliot.dataset.modular_loaders.formats import EmbeddingPayload
from elliot.utils.enums import EntityAxis
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="user_features",
    format="embedding",
    entity_axis={"user_similarity": EntityAxis.USER}
)
class UserUser(AbstractLoader):
    """Square user-user similarity/sentiment matrix, read from a pairwise JSON file
    (see `pairwise_raw_to_embedding_payload` for the supported raw layouts).
    """

    interactions_uu: str

    def __init__(self, **params: Any):
        super().__init__(**params)

        # Initializing variables
        self._raw: Dict[str, Any] = {}

        self._raw = self.reader.read_json(
            path=self.interactions_uu,
            encoding=self._reader_config.encoding
        )

        # Narrow the user domain to ids actually referenced in the similarity file
        self.users = self.users & pairwise_ids_from_raw(self._raw)

    def get_all_features(self, public_users: Dict[Any, int]) -> Tuple[List[int], List[int]]:
        """Return the user-user edge list (row, col) as public user indices.

        Args:
            public_users (Dict[Any, int]): Mapping from user id to public index.

        Returns:
            Tuple[List[int], List[int]]: Parallel (row, col) public index lists, one
                entry per user-user edge.
        """
        rows_uu, cols_uu = [], []

        # Coerce numeric-looking string keys back to int ids before the lookup
        for k, v in self._raw.items():
            for val in v:
                rows_uu.append(public_users[k if not k.isdigit() else int(k)])
                cols_uu.append(public_users[val])

        return rows_uu, cols_uu

    def load(self) -> Dict[str, EmbeddingPayload]:
        """Build the `user_similarity` payload from `self._raw`.

        Returns:
            Dict[str, EmbeddingPayload]: The `user_similarity` payload.
        """
        id_map = public_id_map(self.users)
        payload = pairwise_raw_to_embedding_payload(self._raw, id_map)
        return {"user_similarity": payload}
