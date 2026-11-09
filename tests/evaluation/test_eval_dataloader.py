import pytest
import numpy as np

from elliot.dataset import DataSetLoader
from elliot.evaluation.protocol import EvalProtocol, prepare_protocols
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner, check_path

from tests.params import params_eval_protocol_fail as p
from tests.params import eval_data_folder, eval_dataset_path, user_filter_path

current_path = path_joiner(__file__)


def eval_data(protocol_dict={}, config_dict={}, **kwargs):
    val_data, main_data = _load_data(protocol_dict, config_dict)
    val_dataloader = val_data.protocol.get_eval_dataloader(**kwargs)
    test_dataloader = main_data.protocol.get_eval_dataloader(**kwargs)
    return val_dataloader, test_dataloader, val_data, main_data

def _load_data(protocol_dict={}, config_dict={}):
    config_data = {
        "experiment": {
            "dataset": "eval_dataloader",
            "data_config": {
                "strategy": "dataset",
                "data_folder": eval_data_folder,
                "dataset_path": eval_dataset_path(),
                "reader": {"header": True},
            },
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    "test_ratio": 0.1
                },
                "validation_splitting": {
                    "strategy": "temporal_holdout",
                    "test_ratio": 0.1
                }
            },
            "top_k": 10,
            "evaluation": {
                "simple_metrics": ["nDCG"],
                "protocol": protocol_dict
            },
            **config_dict
        }
    }

    config = build_namespace(config_path=current_path, config_data=config_data)

    loader = DataSetLoader(config=config)
    val_data, main_data = loader.build()

    prepare_protocols(config, val_data, main_data)

    return val_data[0][0], main_data[0]


class TestEvalDataloader:

    @pytest.mark.parametrize("target,expected_targets,expected_contexts,expected_seen", [
        ("all", [
            ["'22'", "'17'", "'11'", "'29'", "'30'"], ["'01'"], ["'06'", "'07'", "'08'", "'09'", "'10'"]
        ], [
            ["'34'", "'37'", "'20'"], ["'01'", "'01'", "'01'"], ["'03'", "'04'", "'05'"]
        ], set()),
        ("first", [
            ["'22'"], ["'01'"], ["'06'"]
        ], [
            ["'34'", "'37'", "'20'"], ["'01'", "'01'", "'01'"], ["'03'", "'04'", "'05'"]
        ], set()),
        ("last", [
            ["'30'"], ["'01'"], ["'10'"]
        ], [
            ["'17'", "'11'", "'29'"], ["'01'", "'01'", "'01'"], ["'07'", "'08'", "'09'"]
        ], {"'06'", "'07'", "'08'", "'09'"}),
    ])
    def test_all_first_last_targets(self, target, expected_targets, expected_contexts, expected_seen):
        config = {
            "target": target
        }

        val_dataloader, test_dataloader, _, main_data = eval_data(config, batch_size=2)

        test_rows = test_dataloader.dataset.rows
        val_rows = val_dataloader.dataset.rows
        _, items = main_data.get_inverse_mappings()

        # One row per user, its context being everything before its first target
        assert test_rows.row_public_ids == ["1", "2", "3"]
        assert test_rows.target_public_ids == expected_targets

        seqs, lens = test_rows.get_eval_context(range(test_rows.n_rows), max_seq_len=3)
        contexts = [[items[i] for i in seq[:n].tolist()] for seq, n in zip(seqs, lens)]
        assert contexts == expected_contexts

        # User 3's seen items hold its train items, plus the test items preceding the target
        _, seen = test_rows.get_seen_mask([2])
        test_items = {"'06'", "'07'", "'08'", "'09'", "'10'"}
        assert {items[i] for i in seen.tolist()} & test_items == expected_seen

        # Validation uses "last" with any target but "all"
        expected_val_targets = (
            [["'31'", "'25'", "'34'", "'37'", "'20'"], ["'01'"], ["'01'", "'02'", "'03'", "'04'", "'05'"]]
            if target == "all" else [["'20'"], ["'01'"], ["'05'"]]
        )
        assert val_rows.row_public_ids == ["1", "2", "3"]
        assert val_rows.target_public_ids == expected_val_targets

    def test_successive_targets(self):
        config = {
            "target": "successive"
        }

        val_dataloader, test_dataloader, _, main_data = eval_data(config, batch_size=2)

        test_rows = test_dataloader.dataset.rows
        val_rows = val_dataloader.dataset.rows
        _, items = main_data.get_inverse_mappings()

        # One row per test item, each user owning several rows
        assert test_rows.multi_row
        assert test_rows.row_public_ids == [
            "1::0", "1::1", "1::2", "1::3", "1::4", "2::0", "3::0", "3::1", "3::2", "3::3", "3::4"
        ]
        assert test_rows.owner_public_ids == ["1"] * 5 + ["2"] + ["3"] * 5
        assert test_rows.target_public_ids == [
            ["'22'"], ["'17'"], ["'11'"], ["'29'"], ["'30'"], ["'01'"],
            ["'06'"], ["'07'"], ["'08'"], ["'09'"], ["'10'"]
        ]

        # Each row's context also holds the test items preceding its target
        seqs, lens = test_rows.get_eval_context([7], max_seq_len=3)
        assert [items[i] for i in seqs[0][:lens[0]].tolist()] == ["'04'", "'05'", "'06'"]

        _, seen = test_rows.get_seen_mask([7])
        seen = {items[i] for i in seen.tolist()}
        assert "'06'" in seen
        assert "'07'" not in seen

        # Validation keeps a single row per user
        assert not val_rows.multi_row
        assert val_rows.row_public_ids == ["1", "2", "3"]
        assert val_rows.target_public_ids == [["'20'"], ["'01'"], ["'05'"]]

    def test_random_targets(self):
        config = {
            "target": "random"
        }

        _, test_dataloader, _, _ = eval_data(config, batch_size=2)
        _, other_test_dataloader, _, _ = eval_data(config, batch_size=2)

        test_rows = test_dataloader.dataset.rows

        assert test_rows.target_public_ids == other_test_dataloader.dataset.rows.target_public_ids

        # Every target comes right after its owner's train items and the test items preceding it
        train_lengths = {"1": 45, "2": 9, "3": 45}
        test_items = {
            "1": ["'22'", "'17'", "'11'", "'29'", "'30'"],
            "2": ["'01'"],
            "3": ["'06'", "'07'", "'08'", "'09'", "'10'"]
        }

        _, lens = test_rows.get_eval_context(range(test_rows.n_rows), max_seq_len=50)
        for owner, (target,), length in zip(
            test_rows.owner_public_ids, test_rows.target_public_ids, lens.tolist()
        ):
            assert length == train_lengths[owner] + test_items[owner].index(target)

    @pytest.mark.parametrize("target", ["all", "last", "successive"])
    def test_ground_truth_is_each_row_targets_rated_as_in_eval_set(self, target):
        config = {
            "target": target
        }

        _, test_dataloader, _, main_data = eval_data(config, batch_size=2)

        test_rows = test_dataloader.dataset.rows
        eval_dict = main_data.eval_set.get_dict()

        assert list(test_rows.ground_truth) == test_rows.row_public_ids
        for row, owner, targets in zip(
            test_rows.row_public_ids, test_rows.owner_public_ids, test_rows.target_public_ids
        ):
            assert test_rows.ground_truth[row] == {item: eval_dict[owner][item] for item in targets}

    def test_every_fold_gets_its_own_protocol(self):
        val_data, main_data = _load_data()

        assert val_data.protocol is not main_data.protocol
        assert val_data.protocol.dataset is val_data
        assert main_data.protocol.dataset is main_data

    def test_fold_standing_in_for_validation_is_prepared_once(self, monkeypatch):
        # With no validation splitting, the test fold also stands in for its missing
        # validation fold: it must be prepared (and its negatives sampled) only once
        config = {
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    "test_ratio": 0.1
                }
            }
        }
        sampled = []
        sample = EvalProtocol.sample_user_negatives
        monkeypatch.setattr(
            EvalProtocol, "sample_user_negatives", lambda self: sampled.append(self) or sample(self)
        )

        protocol = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 5
            }
        }
        val_data, main_data = _load_data(protocol, config)

        assert val_data is main_data
        assert sampled == [main_data.protocol]

    @pytest.mark.parametrize("id_space", ["public", "private"])
    @pytest.mark.parametrize("negative_sampling", [None, {"strategy": "random", "num_negatives": 5}])
    def test_user_filter_keeps_only_filtered_users_rows(self, id_space, negative_sampling):
        config = {
            "target": "successive",
            "user_filter": {
                "path": user_filter_path(id_space),
                "id_space": id_space
            },
            **({"negative_sampling": negative_sampling} if negative_sampling else {})
        }

        val_dataloader, test_dataloader, _, _ = eval_data(config, batch_size=2)

        # Unknown ids are dropped; users 1 and 3 keep all their rows, user 2 none
        test_rows = test_dataloader.dataset.rows
        assert set(test_rows.owner_public_ids) == {"1", "3"}
        assert test_rows.row_public_ids == [
            "1::0", "1::1", "1::2", "1::3", "1::4", "3::0", "3::1", "3::2", "3::3", "3::4"
        ]
        assert set(test_rows.ground_truth) == set(test_rows.row_public_ids)
        assert len(test_dataloader.dataset) == test_rows.n_rows

        # Only the filtered users' rows are ever handed out for scoring
        seen_rows = [r for rows, _ in test_dataloader for r in rows.tolist()]
        assert seen_rows == list(range(test_rows.n_rows))

        assert val_dataloader.dataset.rows.owner_public_ids == ["1", "3"]

    def test_user_filter_without_known_users_keeps_every_user(self):
        config = {
            "user_filter": {
                "path": user_filter_path("unknown")
            }
        }

        _, test_dataloader, _, _ = eval_data(config, batch_size=2)

        assert test_dataloader.dataset.rows.owner_public_ids == ["1", "2", "3"]

    def test_positive_items_aligned_with_user_domain(self):
        config = {
            "splitting": {
                "test_splitting": {
                    "strategy": "fixed_timestamp",
                    "timestamp": 140000
                },
                "validation_splitting": {
                    "strategy": "fixed_timestamp",
                    "timestamp": 100000
                }
            }
        }

        val_data, main_data = _load_data(config_dict=config)

        # User 2's interactions all precede both timestamps: it has no eval item,
        # yet it keeps its (empty) entry
        for data in (val_data, main_data):
            positives = data.eval_set.get_positive_items()
            assert len(positives) == 3
            assert positives[1] == []

    @pytest.mark.parametrize("target", ["all", "first"])
    def test_neg_random(self, target):
        num_negatives = 20
        config = {
            "target": target,
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": num_negatives
            }
        }

        val_dataloader, test_dataloader, _, _ = eval_data(config, batch_size=2)

        train_val_positives = (
            np.array(range(0, 40)), np.array(0), np.array(list(range(0, 5)) + list(range(10, 50)))
        )
        # The test fold's train includes the validation items
        train_test_positives = (
            np.array(range(0, 40)), np.array(0), np.array(range(0, 50))
        )

        # Every target is a candidate: all eval items with "all", a single one
        # otherwise (validation uses "last" with any target but "all")
        expected_val_pos = (5, 1, 5) if target == "all" else (1, 1, 1)
        # At test time user 2 has seen every item (train, validation and test):
        # with no negatives left, its row has no candidates at all
        expected_test_pos = (5, 1, 0) if target == "all" else (1, 1, 0)

        # At validation time the test items are unknown, so they stay sampleable:
        # user 2's only unseen items are its test ones, all picked since
        # num_negatives exceeds the pool
        expected_val_neg = {2: set(range(5, 10))}

        def check_dataloader(dataloader, positives, expected_eval_pos, expected_neg={}):
            for batch in dataloader:
                for i, eval_ in zip(*batch):
                    i, eval_ = i.numpy(), eval_.numpy()
                    eval_ = eval_[eval_ != -1]
                    is_pos = np.isin(eval_, positives[i])
                    num_eval_pos = is_pos.sum()
                    assert num_eval_pos == expected_eval_pos[i]
                    assert len(eval_) - num_eval_pos <= num_negatives
                    if int(i) in expected_neg:
                        assert set(eval_[~is_pos].tolist()) == expected_neg[int(i)]

        check_dataloader(val_dataloader, train_val_positives, expected_val_pos, expected_val_neg)
        check_dataloader(test_dataloader, train_test_positives, expected_test_pos)

    @pytest.mark.parametrize("target", ["all", "first"])
    def test_neg_fixed(self, target):
        config = {
            "target": target,
            "negative_sampling": {
                "strategy": "fixed",
                "read_folder": eval_data_folder
            }
        }

        val_dataloader, test_dataloader, _, _ = eval_data(config, batch_size=2)

        val_negatives = (
            np.array(range(40, 45)), np.array(range(1, 6)), np.array([])
        )
        test_negatives = (
            np.array(range(45, 50)), np.array(range(6, 11)), np.array([])
        )
        expected_val_test_pos = (5, 1, 5) if target == "all" else (1, 1, 1)

        def check_dataloader(dataloader, negatives):
            for batch in dataloader:
                for i, eval_ in zip(*batch):
                    i, eval_ = i.numpy(), eval_.numpy()
                    eval_ = eval_[eval_ != -1]
                    if eval_.size:
                        num_eval_pos = (~np.isin(eval_, negatives[i])).sum()
                        assert num_eval_pos == expected_val_test_pos[i]
                        assert np.isin(negatives[i], eval_).all()
                    else:
                        assert i == 2

        check_dataloader(val_dataloader, val_negatives)
        check_dataloader(test_dataloader, test_negatives)

    def test_neg_saving_on_disk(self):
        save_folder = path_joiner(eval_data_folder, "negative")
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 20,
                "save_on_disk": True,
                "save_folder": save_folder
            }
        }

        eval_data(config, batch_size=2)

        val_path = path_joiner(save_folder.format("eval_dataloader"), "test1_val1_negative.tsv")
        test_path = path_joiner(save_folder.format("eval_dataloader"), "test1_negative.tsv")
        assert check_path(val_path)
        assert check_path(test_path)


class TestEvalDataloaderFailures:

    @pytest.mark.parametrize("params", p["invalid_neg_random"])
    def test_invalid_params_neg_random(self, params):
        config = {
            "negative_sampling": {
                "strategy": "random",
                **params
            }
        }

        with pytest.raises(ValueError):
            eval_data(config, batch_size=2)

    def test_invalid_save_folder_neg_random(self):
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": 10,
                "save_on_disk": True,
                "save_folder": 3
            }
        }

        with pytest.raises(ValueError):
            eval_data(config, batch_size=2)

    @pytest.mark.parametrize("params", p["invalid_neg_fixed"])
    def test_invalid_or_missing_params_neg_fixed(self, params):
        config = {
            "negative_sampling": {
                "strategy": "fixed",
                **({"read_folder": params["read_folder"]} if params["read_folder"] is not None else {})
            }
        }

        with pytest.raises((FileNotFoundError, ValueError, AttributeError)):
            eval_data(config, batch_size=2)

    @pytest.mark.parametrize("params", p["invalid_strategy"])
    def test_invalid_or_missing_strategy(self, params):
        config = {
            "negative_sampling": {
                **({"strategy": params["strategy"]} if params["strategy"] is not None else {}),
                "num_negatives": 10
            }
        }

        with pytest.raises(ValueError):
            eval_data(config, batch_size=2)

    @pytest.mark.parametrize("params", p["missing_user_filter_file"])
    def test_missing_user_filter_file(self, params):
        config = {
            "user_filter": params
        }

        with pytest.raises(FileNotFoundError):
            eval_data(config, batch_size=2)

    @pytest.mark.parametrize("params", p["invalid_user_filter_id_space"])
    def test_invalid_user_filter_id_space(self, params):
        config = {
            "user_filter": {"path": "users.tsv", **params}
        }

        with pytest.raises(ValueError):
            eval_data(config, batch_size=2)


if __name__ == "__main__":
    pytest.main()
