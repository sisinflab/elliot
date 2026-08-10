from typing import Dict, Tuple, Optional
from ast import literal_eval

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader
from elliot.dataset.modular_loaders.build import npy_folder_to_embedding_payload, public_id_map
from elliot.dataset.modular_loaders.materialize import embedding_to_dense
from elliot.dataset.modular_loaders.formats import EmbeddingPayload
from elliot.utils.enums import AlignmentMode, EntityAxis, Materialization
from elliot.utils.registry import side_info_registry


@side_info_registry.register(
    provides="item_features",
    format="embedding",
    alignment=AlignmentMode.PAD,
    materialization=Materialization.MMAP,
    entity_axis={
        "visual_features": EntityAxis.ITEM,
        "visual_pca_features": EntityAxis.ITEM,
        "visual_feat_map_features": EntityAxis.ITEM,
    }
)
class VisualAttribute(AbstractLoader):
    """One or more precomputed dense feature vectors per item -- raw, PCA-reduced,
    and/or feature-map -- each read from its own folder of `.npy` files (filename
    stem = item id). Any combination of the three folders may be configured; only
    those found are returned by `load()`.
    """

    visual_feature_folder_path: Optional[str] = None
    visual_pca_feature_folder_path: Optional[str] = None
    visual_feat_map_feature_folder_path: Optional[str] = None
    images_folder_path: Optional[str] = None
    image_size_tuple: Optional[str] = None

    def __init__(self, **params):
        super().__init__(**params)

        # Initializing variables
        self._visual_features_shape: Optional[int] = None
        self._visual_pca_features_shape: Optional[int] = None
        self._visual_feat_map_features_shape: Optional[Tuple[int, ...]] = None
        self._item_mapping: Dict[int, int] = {}

        if self.image_size_tuple is not None:
            self.image_size_tuple = literal_eval(self.image_size_tuple)

        items = set()

        folder_items, _, shape = self.reader.discover_npy_ids(
            folder_path=self.visual_feature_folder_path
        )
        items |= folder_items
        if shape:
            self._visual_features_shape = shape[0]

        folder_items, _, shape = self.reader.discover_npy_ids(
            folder_path=self.visual_pca_feature_folder_path
        )
        items |= folder_items
        if shape:
            self._visual_pca_features_shape = shape[0]

        folder_items, _, shape = self.reader.discover_npy_ids(
            folder_path=self.visual_feat_map_feature_folder_path
        )
        items |= folder_items
        if shape:
            self._visual_feat_map_features_shape = shape

        folder_items, _, _ = self.reader.discover_npy_ids(
            folder_path=self.images_folder_path
        )
        items |= folder_items

        if items:
            self._item_mapping = {item: idx for idx, item in enumerate(sorted(items))}

        self.items = self.items & items

    def get_all_features(self):
        return self.get_all_visual_features()

    def get_all_visual_features(self):
        """Dense `(n_items, dim)` matrix of raw visual features, read through `load()`
        so it honors `self.materialization` (and the correctly re-indexed, post-filter
        `id_map` it builds) exactly like every other consumer of this loader's payload,
        rather than re-reading files by hand against the *pre-filter* `item_mapping`.
        """
        return embedding_to_dense(self.load()["visual_features"])

    def get_all_visual_pca_features(self):
        return embedding_to_dense(self.load()["visual_pca_features"])

    def get_all_visual_feat_map_features(self):
        return embedding_to_dense(self.load()["visual_feat_map_features"])

    def load(self) -> Dict[str, EmbeddingPayload]:
        """Materialize the configured visual feature source(s) (raw/PCA/feature-map)
        into `EmbeddingPayload`s, one named entry per configured folder, via the shared
        `npy_folder_to_embedding_payload` adapter. When this loader's `materialization`
        is `LAZY` (read-and-copy on demand) or `MMAP` (memory-mapped on demand, the
        registered default), each payload exposes a `row_loader` instead of a fully
        materialized matrix, so a consumer like VBPR/VNPR/DeepStyle/ACF/DVBPR can keep
        reading one `.npy` file per item on demand during batch sampling; with `MEMORY`
        each payload is a fully materialized dense matrix instead. `get_all_visual_*`
        go through this same method (via `embedding_to_dense`), so they honor whatever
        `materialization` is configured too.
        """
        id_map = public_id_map(i for i in self._item_mapping if i in self.items)

        def make_payload(folder_path: Optional[str], shape) -> Optional[EmbeddingPayload]:
            if not folder_path or not id_map:
                return None
            row_shape = shape if isinstance(shape, tuple) else (shape,)
            return npy_folder_to_embedding_payload(
                folder_path, id_map, row_shape, self.materialization, self.reader
            )

        payloads = {}
        visual = make_payload(
            self.visual_feature_folder_path, self._visual_features_shape
        )
        if visual is not None:
            payloads["visual_features"] = visual

        pca = make_payload(
            self.visual_pca_feature_folder_path, self._visual_pca_features_shape
        )
        if pca is not None:
            payloads["visual_pca_features"] = pca

        feat_map = make_payload(
            self.visual_feat_map_feature_folder_path, self._visual_feat_map_features_shape
        )
        if feat_map is not None:
            payloads["visual_feat_map_features"] = feat_map

        return payloads
