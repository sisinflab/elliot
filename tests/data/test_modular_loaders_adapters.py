import pytest
import numpy as np

from elliot.dataset.modular_loaders.build import (
    npy_folder_to_embedding_payload,
    pairwise_ids_from_raw,
    pairwise_raw_to_embedding_payload,
    rows_to_embedding_payload
)
from elliot.dataset.modular_loaders.materialize import (
    embedding_to_dense,
    embedding_to_sparse,
    feature_map_to_sparse,
    graph_triples_to_adjacency,
    graph_triples_to_edge_index,
    text_to_padded_ids
)
from elliot.dataset.modular_loaders.remap import (
    remap_embedding_payload,
    remap_pair_payload,
    remap_text_payload
)
from elliot.dataset.modular_loaders.formats import EmbeddingPayload, GraphPayload, TextPayload
from elliot.utils.enums import Materialization
from elliot.utils.folder import path_joiner

from tests.params import data_folder


class TestAdapters:

    def test_embedding_to_dense_from_sparse(self):
        import scipy.sparse as sp

        sparse = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        payload = EmbeddingPayload(sparse=sparse, row_ids=[10, 20], id_map={10: 0, 20: 1})
        dense = embedding_to_dense(payload, ids=[20, 10])
        assert np.array_equal(dense, np.array([[0.0, 1.0], [1.0, 0.0]]))

    def test_embedding_to_dense_from_row_loader(self):
        vectors = {0: np.array([1.0, 2.0]), 1: np.array([3.0, 4.0])}
        payload = EmbeddingPayload(row_loader=lambda idx: vectors[idx], row_ids=[10, 11], id_map={10: 0, 11: 1})
        dense = embedding_to_dense(payload)
        assert np.array_equal(dense, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_embedding_to_sparse_from_dense(self):
        payload = EmbeddingPayload(dense=np.array([[1.0, 0.0]]))
        sparse = embedding_to_sparse(payload)
        assert sparse.shape == (1, 2)

    def test_feature_map_to_sparse(self):
        feature_map = {1: [0, 2], 2: [1]}
        id_map = {1: 0, 2: 1}
        sparse = feature_map_to_sparse(feature_map, id_map, n_cols=3)
        assert sparse.shape == (2, 3)
        assert sparse.toarray().tolist() == [[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]

    def test_text_to_padded_ids(self):
        payload = TextPayload(tokens={1: [5, 6, 7], 2: [8]}, id_map={1: 0, 2: 1})
        padded, lengths = text_to_padded_ids(payload, max_len=4)
        assert padded.shape == (2, 4)
        assert lengths.tolist() == [3, 1]
        assert padded[0].tolist() == [5, 6, 7, 0]

    def test_graph_triples_to_edge_index_and_adjacency(self):
        payload = GraphPayload(
            heads=np.array([0, 1]),
            relations=np.array([1, 1]),
            tails=np.array([1, 2]),
            entity2id={0: 0, 1: 1, 2: 2},
            relation2id={10: 1},
        )
        edge_index = graph_triples_to_edge_index(payload)
        assert edge_index.shape == (2, 2)
        adjacency = graph_triples_to_adjacency(payload, n_nodes=3)
        assert adjacency.nnz == 4  # symmetrized

    def test_pairwise_ids_from_raw_adjacency_list_layout(self):
        assert pairwise_ids_from_raw({"1": ["2", "3"]}) == {1, 2, 3}

    def test_pairwise_ids_from_raw_weighted_pair_key_layout(self):
        assert pairwise_ids_from_raw({"1_2": 0.5, "2_3": 0.75}) == {1, 2, 3}


class TestRowsToEmbeddingPayload:

    def _row_reader_and_calls(self):
        calls = []

        def row_reader(entity_id, mmap_mode):
            calls.append((entity_id, mmap_mode))
            return np.array([entity_id, entity_id * 2], dtype=np.float32)

        return row_reader, calls

    def test_memory_calls_row_reader_with_no_mmap_mode_for_every_id_upfront(self):
        row_reader, calls = self._row_reader_and_calls()
        id_map = {10: 0, 20: 1}

        payload = rows_to_embedding_payload(Materialization.MEMORY, id_map, (2,), row_reader)

        assert payload.row_loader is None
        assert np.array_equal(payload.dense, np.array([[10.0, 20.0], [20.0, 40.0]]))
        assert set(calls) == {(10, None), (20, None)}

    def test_lazy_defers_calls_with_no_mmap_mode(self):
        row_reader, calls = self._row_reader_and_calls()
        id_map = {10: 0, 20: 1}

        payload = rows_to_embedding_payload(Materialization.LAZY, id_map, (2,), row_reader)

        assert payload.dense is None
        assert calls == []  # nothing read until a row is actually requested
        assert np.array_equal(payload.row_loader(1), np.array([20.0, 40.0]))
        assert calls == [(20, None)]

    def test_mmap_defers_calls_with_r_mmap_mode(self):
        row_reader, calls = self._row_reader_and_calls()
        id_map = {10: 0, 20: 1}

        payload = rows_to_embedding_payload(Materialization.MMAP, id_map, (2,), row_reader)

        assert payload.dense is None
        assert calls == []
        payload.row_loader(0)
        assert calls == [(10, "r")]


class TestNpyFolderToEmbeddingPayloadMaterialization:
    data_path = path_joiner(data_folder.format("modular_loaders"), "npy_folder")
    id_map = {1: 0, 2: 1}

    def test_memory_returns_plain_dense_array(self):
        payload = npy_folder_to_embedding_payload(self.data_path, self.id_map, (2,), Materialization.MEMORY)

        assert payload.row_loader is None
        assert payload.dense is not None
        assert not isinstance(payload.dense, np.memmap)
        assert np.array_equal(payload.dense, np.array([[1.0, 2.0], [3.0, 4.0]]))

    def test_lazy_returns_row_loader_with_plain_copies(self):
        payload = npy_folder_to_embedding_payload(self.data_path, self.id_map, (2,), Materialization.LAZY)

        assert payload.dense is None
        assert payload.row_loader is not None
        row = payload.row_loader(0)
        assert not isinstance(row, np.memmap)
        assert np.array_equal(row, np.array([1.0, 2.0]))

    def test_mmap_returns_row_loader_with_memmapped_rows(self):
        payload = npy_folder_to_embedding_payload(self.data_path, self.id_map, (2,), Materialization.MMAP)

        assert payload.dense is None
        assert payload.row_loader is not None
        row = payload.row_loader(0)
        assert isinstance(row, np.memmap)
        assert np.array_equal(row, np.array([1.0, 2.0]))


class TestRemapEmbeddingPayload:

    def test_reindexes_dense_rows_to_private_order(self):
        payload = EmbeddingPayload(
            dense=np.array([[10.0], [20.0], [30.0], [40.0]]),
            row_ids=[10, 20, 30, 40],
            id_map={10: 0, 20: 1, 30: 2, 40: 3},
        )
        # Fold covers a reordered subset: private id 0 -> public 30, private id 1 -> public 10.
        view = remap_embedding_payload(payload, inv_mapping=[30, 10])

        assert view.dense.tolist() == [[30.0], [10.0]]
        assert view.id_map == {0: 0, 1: 1}
        assert view.row_ids == [0, 1]

    def test_wraps_row_loader_without_calling_it_eagerly(self):
        calls = []

        def row_loader(row, _calls=calls):
            _calls.append(row)
            return np.array([row * 10.0])

        payload = EmbeddingPayload(row_loader=row_loader, row_ids=[10, 20], id_map={10: 0, 20: 1}, shape=(2, 1))
        view = remap_embedding_payload(payload, inv_mapping=[20, 10])

        assert calls == []  # nothing read yet
        assert view.row_loader(0).tolist() == [10.0]  # private 0 -> public 20 -> row 1
        assert calls == [1]

    def test_remaps_square_pairwise_payload_on_both_axes(self):
        # item-item: public 1<->2 similar, public 3 unrelated.
        raw = {"1": [2], "2": [1]}
        id_map = {1: 0, 2: 1, 3: 2}
        payload = pairwise_raw_to_embedding_payload(raw, id_map)

        view = remap_embedding_payload(payload, inv_mapping=[3, 1, 2])

        dense = view.sparse.toarray()
        assert dense.shape == (3, 3)
        assert dense[1, 2] == 1.0  # private 1 (public 1) <-> private 2 (public 2)
        assert dense[0].sum() == 0  # private 0 (public 3) has no similarities

    def test_raises_on_fold_id_missing_from_loader_domain(self):
        payload = EmbeddingPayload(dense=np.array([[1.0]]), row_ids=[10], id_map={10: 0})
        with pytest.raises(KeyError):
            remap_embedding_payload(payload, inv_mapping=[10, 999])


class TestRemapTextPayload:

    def test_reindexes_tokens_to_private_order(self):
        payload = TextPayload(tokens={10: [1, 2], 20: [3]}, id_map={10: 0, 20: 1}, vocab_size=100)
        view = remap_text_payload(payload, inv_mapping=[20, 10])
        assert view.tokens == {0: [3], 1: [1, 2]}
        assert view.id_map == {0: 0, 1: 1}

    def test_raises_on_fold_id_missing_from_loader_domain(self):
        payload = TextPayload(tokens={10: [1]}, id_map={10: 0})
        with pytest.raises(KeyError):
            remap_text_payload(payload, inv_mapping=[10, 20])


class TestRemapPairPayload:

    def test_rekeys_pairs_and_drops_pairs_outside_the_fold(self):
        payload = EmbeddingPayload(
            dense=np.array([[1.0], [2.0]]),
            row_ids=[(100, 1), (100, 2)],
            id_map={(100, 1): 0, (100, 2): 1},
        )
        # This fold only knows about item 1 (item 2 belongs to some other fold), and
        # remaps user 100 -> private 0, item 1 -> private 0.
        view = remap_pair_payload(payload, u_map={100: 0}, i_map={1: 0})

        assert view.id_map == {(0, 0): 0}
        assert view.row_ids == [(0, 0)]
        assert view.dense is payload.dense  # rows themselves are never copied/reordered


if __name__ == '__main__':
    pytest.main()
