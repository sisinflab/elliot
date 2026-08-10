import pytest

from elliot.dataset import DataSetLoader
from elliot.dataset.modular_loaders.materialize import embedding_to_dense, text_to_padded_ids
from elliot.namespace import build_namespace
from elliot.utils.enums import Materialization
from elliot.utils.folder import path_joiner

from tests.params import data_folder, dataset_path

current_path = path_joiner(__file__)


def get_loader(side_information):
    config_data = {
        "experiment": {
            "dataset": "modular_loaders",
            "data_config": {
                "strategy": "dataset",
                "dataset_path": dataset_path(),
                "reader": {
                    "header": True,
                    "dtypes": {"user_id_type": "int", "item_id_type": "int"}
                },
                "side_information": side_information,
            },
            "splitting": {
                "test_splitting": {
                    "strategy": "random_subsampling",
                    "test_ratio": 0.2
                }
            }
        }
    }
    config = build_namespace(config_path=current_path, config_data=config_data)
    return DataSetLoader(config=config)

def load_data(side_information, before_first_load=None):
    name = side_information["dataloader"]
    loader = get_loader(side_information)
    val_data, main_data = loader.build()
    loader.prepare_dataset(val_data, main_data)
    train_set = main_data[0].train_set

    if before_first_load is not None:
        before_first_load(train_set.get_loader(name))

    return train_set.get_side_info(name), train_set, loader


class TestItemAttributes:
    data_path = path_joiner(data_folder, "item_attributes")

    def test_returns_item_features_aligned_to_the_train_fold(self):
        config = {
            "dataloader": "ItemAttributes",
            "attribute_file": path_joiner(self.data_path, "attributes.tsv"),
        }

        payloads, train_set, _ = load_data(config)

        payload = payloads["item_features"]
        assert payload.shape[0] == train_set.dims[1]
        assert len(payload.col_ids) > 0
        assert embedding_to_dense(payload).sum() > 0

    def test_load_narrows_items_to_attribute_file_coverage(self):
        config = {
            "dataloader": "ItemAttributes",
            "attribute_file": path_joiner(self.data_path, "attrs.tsv"),
        }

        loader = get_loader(config)

        assert loader.side_information["ItemAttributes"].get_mapped()[1] == {1}


class TestPairwiseLoaders:
    data_path = path_joiner(data_folder, "pairwise")

    @pytest.mark.parametrize("name,key_name,field_name,file_name,dim", [
        ("ItemItem", "item_similarity", "interactions_ii", "item_item.json", 1),
        ("UserUser", "user_similarity", "interactions_uu", "user_user.json", 0),
    ])
    def test_returns_square_sparse_payload_aligned_to_the_train_fold(
        self, name, key_name, field_name, file_name, dim
    ):
        config = {
            "dataloader": name,
            field_name: path_joiner(self.data_path, file_name),
        }

        payloads, train_set, _ = load_data(config)

        payload = payloads[key_name]
        assert payload.shape[0] == payload.shape[1] == train_set.dims[dim]
        assert payload.sparse.nnz > 0

    @pytest.mark.parametrize("name,key_name,field_name", [
        ("ItemItem", "item_similarity", "interactions_ii"),
        ("UserUser", "user_similarity", "interactions_uu"),
    ])
    def test_load_supports_weighted_pair_key_layout(self, name, key_name, field_name):
        config = {
            "dataloader": name,
            field_name: path_joiner(self.data_path, "sentiment.json"),
        }
        
        loader = get_loader(config)
        payload = loader.side_information[name].load()[key_name]
        dense = embedding_to_dense(payload)

        a, b, c = (1, 2, 3)
        assert dense[payload.id_map[a], payload.id_map[b]] == pytest.approx(0.75)
        assert dense[payload.id_map[b], payload.id_map[c]] == pytest.approx(0.5)


class TestChainedKG:
    data_path = path_joiner(data_folder, "chained_kg")

    def test_load_via_shared_feature_map_adapter(self):
        config = {
            "dataloader": "ChainedKG",
            "map": path_joiner(self.data_path, "map.tsv"),
            "features": path_joiner(self.data_path, "features.tsv"),
            "properties": path_joiner(self.data_path, "properties.txt"),
            "additive": True,
            "threshold": 0,
        }

        payloads, train_set, _ = load_data(config)

        chainedkg = train_set.get_loader("ChainedKG")
        assert chainedkg.alignment is not None
        assert chainedkg.get_mapped()[1]  # non-empty, real item domain

        payload = payloads["item_features"]
        assert payload.shape[0] == train_set.dims[1]
        assert len(payload.col_ids) > 0

        # Second request returns the exact same cached dict (no reload) -- the general
        # caching contract itself is covered end-to-end in
        # `TestInteractionsPrivateSideInfoView`.
        assert train_set.get_side_info("ChainedKG") is payloads


class TestKAHFMLoader:
    data_path = path_joiner(data_folder, "kahfm")

    def test_load_derives_features_from_raw_triples(self):
        config = {
            "dataloader": "KAHFMLoader",
            "mapping": path_joiner(self.data_path, "mapping.tsv"),
            "kg_train": path_joiner(self.data_path, "train.tsv"),
            "additive": True,
            "threshold": 1.0,
        }

        payloads, train_set, _ = load_data(config)

        payload = payloads["item_features"]
        assert payload.shape[0] == train_set.dims[1]
        assert len(payload.col_ids) > 0


class TestKGFlexLoader:
    data_path = path_joiner(data_folder, "kgflex")

    def test_load_combines_first_and_second_order_features(self):
        config = {
            "dataloader": "KGFlexLoader",
            "mapping": path_joiner(self.data_path, "mapping.tsv"),
            "kg_train": path_joiner(self.data_path, "train.tsv"),
            "second_hop": path_joiner(self.data_path, "second_hop.tsv"),
            "additive": True,
            "threshold": 0,
        }

        payloads, train_set, _ = load_data(config)

        payload = payloads["item_features"]
        assert payload.shape[0] == train_set.dims[1]
        assert len(payload.col_ids) > 0


class TestKGCompletion:
    data_path = path_joiner(data_folder, "kg_completion")

    def test_load_returns_graph_payload_and_item_entity_map(self):
        config = {
            "dataloader": "KGCompletion",
            "train_path": path_joiner(self.data_path, "train.txt"),
            "input_type": "standard",
            "mapping": path_joiner(self.data_path, "mapping.tsv"),
        }

        loader = get_loader(config)
        kg_completion = loader.side_information["KGCompletion"]
        payload = kg_completion.load()["kg_triples"]

        assert len(payload.heads) > 100
        assert payload.item_entity_map
        assert set(payload.item_entity_map.keys()) <= kg_completion.items

    def test_load_without_mapping_has_no_item_entity_map(self):
        config = {
            "dataloader": "KGCompletion",
            "train_path": path_joiner(self.data_path, "train_no_map.txt"),
            "input_type": "standard"
        }

        loader = get_loader(config)
        payload = loader.side_information["KGCompletion"].load()["kg_triples"]

        assert payload.item_entity_map is None


class TestKGINTSVLoader:
    data_path = path_joiner(data_folder, "kgin_tsv")

    def test_load_returns_canonical_graph_payload(self):
        config = {
            "dataloader": "KGINTSVLoader",
            "attribute_file": path_joiner(self.data_path, "kg.tsv")
        }

        payloads, _, loader = load_data(config)
        kgin = loader.side_information["KGINTSVLoader"]

        payload = payloads["kg_triples"]
        assert len(payload.heads) > 100
        assert set(payload.item_entity_map.keys()) == kgin.items
        assert len(kgin.items) > 40

    def test_defaults_to_memory_materialization(self):
        config = {
            "dataloader": "KGINTSVLoader",
            "attribute_file": path_joiner(self.data_path, "kg.tsv")
        }

        loader = get_loader(config)

        assert loader.side_information["KGINTSVLoader"].materialization == Materialization.MEMORY


class TestInteractionsTextualAttributes:
    data_path = path_joiner(data_folder, "interactions_textual")

    def test_load_keys_rows_by_user_item_pair(self):
        config = {
            "dataloader": "InteractionsTextualAttributes",
            "interaction_features": path_joiner(self.data_path, "features"),
            "interaction_ids": path_joiner(self.data_path, "interactions.tsv"),
        }

        payloads, _, _ = load_data(config)

        payload = payloads["interaction_features"]
        assert len(payload.row_ids) > 0
        assert embedding_to_dense(payload).shape[1] == 8

    def test_materialization_memory_override_returns_dense_payload(self):
        config = {
            "dataloader": "InteractionsTextualAttributes",
            "interaction_features": path_joiner(self.data_path, "features"),
            "interaction_ids": path_joiner(self.data_path, "interactions.tsv"),
        }
        before_first_load = lambda loader: setattr(loader, "materialization", Materialization.MEMORY)
    
        payloads, _, _ = load_data(config, before_first_load)

        payload = payloads["interaction_features"]
        assert payload.row_loader is None
        assert payload.dense is not None


class TestTextualAttribute:
    data_path = path_joiner(data_folder, "textual_attribute")

    def test_load_returns_item_features_aligned_to_the_train_fold(self):
        config = {
            "dataloader": "TextualAttribute",
            "textual_features": self.data_path,
        }

        payloads, train_set, _ = load_data(config)
        dense = embedding_to_dense(payloads["item_features"])

        assert dense.shape == (train_set.dims[1], 8)

    def test_materialization_memory_override_returns_dense_payload(self):
        config = {
            "dataloader": "TextualAttribute",
            "textual_features": self.data_path,
        }
        before_first_load=lambda loader: setattr(loader, "materialization", Materialization.MEMORY)

        payloads, _, _ = load_data(config, before_first_load)

        payload = payloads["item_features"]
        assert payload.row_loader is None
        assert payload.dense is not None


class TestVisualAttribute:
    data_path = path_joiner(data_folder, "visual_attribute")

    def test_load_uses_row_loader_for_mmap_default(self):
        config = {
            "dataloader": "VisualAttribute",
            "visual_feature_folder_path": self.data_path,
        }

        payloads, train_set, _ = load_data(config)

        payload = payloads["visual_features"]
        assert payload.dense is None
        assert payload.row_loader is not None
        assert embedding_to_dense(payload).shape == (train_set.dims[1], 16)

    def test_materialization_memory_override_returns_dense_payload(self):
        config = {
            "dataloader": "VisualAttribute",
            "visual_feature_folder_path": self.data_path,
        }
        before_first_load=lambda loader: setattr(loader, "materialization", Materialization.MEMORY)

        payloads, _, _ = load_data(config, before_first_load)

        payload = payloads["visual_features"]
        assert payload.row_loader is None
        assert payload.dense is not None


class TestWordsTextualAttributes:
    data_path = path_joiner(data_folder, "words_textual")

    def test_load_returns_tokens_and_vocab_embeddings(self):
        config = {
            "dataloader": "WordsTextualAttributes",
            "users_vocabulary_features": path_joiner(self.data_path, "users_vocab.npy"),
            "items_vocabulary_features": path_joiner(self.data_path, "items_vocab.npy"),
            "users_tokens": path_joiner(self.data_path, "users_tokens.json"),
            "items_tokens": path_joiner(self.data_path, "items_tokens.json")
        }

        payloads, _, loader = load_data(config)
        words = loader.side_information["WordsTextualAttributes"]

        assert set(payloads.keys()) == {"tokens", "users_vocab_embeddings", "items_vocab_embeddings"}
        assert payloads["users_vocab_embeddings"].shape == (60, 8)
        assert payloads["items_vocab_embeddings"].shape == (60, 8)
        # `tokens` merges user *and* item ids into one flat id space (see this
        # loader's own class docstring) -- since this fixture's user ids (1..20) and
        # item ids (1..50) overlap numerically, like they do in the rest of
        # `tests/data/modular_loaders/`, the merged key count is the *union* of the
        # loader's own (post cross-loader-intersection) domains, not their sum.
        assert len(payloads["tokens"].tokens) == len(words.users | words.items)
        padded, _ = text_to_padded_ids(payloads["tokens"], max_len=3)
        assert padded.shape[0] == len(words.users | words.items)


class TestCrossLoaderIntersection:
    data_path = path_joiner(data_folder, "cross_loader_intersection")

    def test_final_universe_is_intersection_of_every_loader(self):
        config = [
            {
                "dataloader": "ItemAttributes",
                "attribute_file": path_joiner(self.data_path, "attrs.tsv")
            },
            {
                "dataloader": "ChainedKG",
                "map": path_joiner(self.data_path, "map.tsv"),
                "features": path_joiner(self.data_path, "features.tsv"),
                "properties": path_joiner(self.data_path, "properties.txt"),
                "threshold": 0,
            }
        ]

        loader = get_loader(config)

        # Neither loader alone would produce this: ItemAttributes covers 1..30,
        # ChainedKG covers 21..50 -- only the 21..30 overlap should survive.
        assert set(loader.dataframe["itemId"].unique()) == set(range(21, 31))
        assert loader.side_information["ItemAttributes"].get_mapped()[1] == set(range(21, 31))
        assert loader.side_information["ChainedKG"].get_mapped()[1] == set(range(21, 31))


if __name__ == '__main__':
    pytest.main()
