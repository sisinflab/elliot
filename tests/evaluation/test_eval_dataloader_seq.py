import pytest

from elliot.dataset import DataSetLoader
from elliot.evaluation.protocol import prepare_protocols
from elliot.namespace import build_namespace
from elliot.utils.enums import SessionStrategy
from elliot.utils.folder import path_joiner

from tests.params import params_eval_protocol_fail as p
from tests.params import data_folder, dataset_path

current_path = path_joiner(__file__)


def eval_data(protocol_dict={}, config_dict={}, load_as_session_only=False, **kwargs):
    val_data, main_data = _load_data(protocol_dict, config_dict, load_as_session_only=load_as_session_only)
    val_dataloader = val_data.protocol.get_eval_dataloader(**kwargs)
    test_dataloader = main_data.protocol.get_eval_dataloader(**kwargs)
    return val_dataloader, test_dataloader, val_data, main_data

def _load_data(protocol_dict={}, config_dict={}, load_as_session_only=False):
    config_data = {
        "experiment": {
            "dataset": "dataloader_sessions",
            "data_config": {
                "strategy": "dataset",
                "data_folder": data_folder,
                "dataset_path": dataset_path(),
                "session_strategy": "session_only" if load_as_session_only else "flat",
                "reader": {"header": True},
            },
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    "leave_n_out": 2
                },
                "validation_splitting": {
                    "strategy": "temporal_holdout",
                    "leave_n_out": 1 if load_as_session_only else 2
                }
            },
            "top_k": 10,
            "evaluation": {
                "protocol": {
                    **({"target": "last"} if load_as_session_only else {}),
                    **protocol_dict
                }
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

    def test_session_only_full_eval_rows(self):
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 1
        }

        val_dataloader, test_dataloader, _, _ = eval_data(load_as_session_only=True, **params)

        def check_seen_users(dataloader, expected_users):
            seen_users = []
            for users, eval_items in dataloader:
                assert eval_items is None
                seen_users.extend(users.tolist())
            assert sorted(seen_users) == expected_users

        check_seen_users(test_dataloader, [0, 1, 2])
        check_seen_users(val_dataloader, [0, 1])

        test_sessions = test_dataloader.dataset.rows
        val_sessions = val_dataloader.dataset.rows

        # User 1's newer test session [1] has no context before its only item,
        # so it yields no row at all
        owner = test_sessions.owner_users
        assert owner is not None
        assert list(owner) == [0, 1, 1]

        row_public_ids = test_sessions.row_public_ids
        assert row_public_ids == ["1::0", "2::0", "2::1"]

        # userId is read back as string by the loader's default dtype config,
        # so the owner map's values are the raw (string) public user ids.
        owner_map = test_sessions.owner_map
        assert owner_map == {"1::0": "1", "2::0": "2", "2::1": "2"}

        # Each row's ground truth is that session's own masked (last) item,
        # never shared with another session of the same owner.
        assert test_sessions.target_public_ids == [["5"], ["6"], ["6"]]

        # Validation has exactly 1 session/user, so unlike the test fold above
        # there's no owner with more than one row to disambiguate.
        owner = val_sessions.owner_users
        assert owner is not None
        assert list(owner) == [0, 1]

        row_public_ids = val_sessions.row_public_ids
        assert row_public_ids == ["1::0", "2::0"]

        owner_map = val_sessions.owner_map
        assert owner_map == {"1::0": "1", "2::0": "2"}

        assert val_sessions.target_public_ids == [["3"], ["6"]]

    def test_session_only_context_is_leave_last_item_out(self):
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 10
        }

        val_dataloader, test_dataloader, _, _ = eval_data(load_as_session_only=True, **params)

        test_sessions = test_dataloader.dataset.rows
        val_sessions = val_dataloader.dataset.rows

        _, lens = test_sessions.get_eval_context([0, 1, 2], max_seq_len=3)

        # row 0: user 1's older test session [4, 5] -> leave-last-out context [4], len 1
        # (user 1's newer test session [1] has no context, hence no row)
        # rows 1/2: user 2's two test sessions [2, 4, 6] each -> context [2, 4], len 2
        assert lens.tolist() == [1, 2, 2]

        _, val_lens = val_sessions.get_eval_context([0, 1], max_seq_len=3)

        # row 0: user 1's validation session [1, 2, 3] -> leave-last-out context [1, 2], len 2
        # row 1: user 2's validation session [2, 4, 6] -> leave-last-out context [2, 4], len 2
        assert val_lens.tolist() == [2, 2]

    def test_session_only_neg_random(self):
        num_negatives = 20
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": num_negatives
            }
        }
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 10
        }

        val_dataloader, test_dataloader, _, _ = eval_data(config, load_as_session_only=True, **params)

        # Test:
        # user 1 -> private 0 with test row 0, session [4, 5], target 5 -> private 4;
        # user 2 -> private 1 with test rows 1 and 2, both session [2, 4, 6], target 6 -> private 5.
        target = {0: 4, 1: 5, 2: 5}
        owner_of = {0: 0, 1: 1, 2: 1}

        # Allowed negatives: every item never seen by that owning user,
        # in train or in *any* of their eval sessions
        allowed_negatives = {
            0: {5},        # user 1 has seen private items {0, 1, 2, 3, 4} (train + both test sessions)
            1: {0, 2, 4},  # user 2 has seen private items {1, 3, 5} (train + both test sessions)
        }

        def check_dataloader(dataloader, expected_rows, target, owner_of):
            seen_rows = set()
            for users, eval_items in dataloader:
                for row, eval_ in zip(users.tolist(), eval_items):
                    eval_ = eval_[eval_ != -1].tolist()
                    seen_rows.add(row)

                    # SESSION_ONLY has exactly one ground-truth positive per
                    # row (that session's own masked/last item)
                    expected = allowed_negatives[owner_of[row]] | {target[row]}
                    assert set(eval_) == expected

            assert seen_rows == expected_rows

        check_dataloader(test_dataloader, {0, 1, 2}, target, owner_of)

        # Validation:
        # user 1 -> private 0 with val row 0, session [1, 2, 3], target 3 -> private 2;
        # user 1 -> private 0 with val row 1, session [2, 4, 6], target 6 -> private 5;
        val_target = {0: 2, 1: 5}
        val_owner_of = {0: 0, 1: 1}

        check_dataloader(val_dataloader, {0, 1}, val_target, val_owner_of)

    def test_session_only_successive_targets(self):
        config = {
            "target": "successive"
        }
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 10
        }

        val_dataloader, test_dataloader, _, _ = eval_data(config, load_as_session_only=True, **params)

        test_sessions = test_dataloader.dataset.rows
        val_sessions = val_dataloader.dataset.rows

        # One row per session item but the first: user 1's older test session [4, 5]
        # yields one row, each of user 2's two test sessions [2, 4, 6] yields two
        assert test_sessions.row_public_ids == ["1::0", "2::0", "2::1", "2::2", "2::3"]
        assert test_sessions.target_public_ids == [["5"], ["4"], ["6"], ["4"], ["6"]]

        # Each row's context is the session prefix before its target
        _, lens = test_sessions.get_eval_context(range(test_sessions.n_rows), max_seq_len=3)
        assert lens.tolist() == [1, 1, 2, 1, 2]

        # Validation keeps a single row per session, targeting its last item
        assert val_sessions.row_public_ids == ["1::0", "2::0"]
        assert val_sessions.target_public_ids == [["3"], ["6"]]

    def test_session_only_random_targets(self):
        config = {
            "target": "random"
        }
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 10
        }

        _, test_dataloader, _, _ = eval_data(config, load_as_session_only=True, **params)
        _, other_test_dataloader, _, _ = eval_data(config, load_as_session_only=True, **params)

        test_sessions = test_dataloader.dataset.rows

        assert test_sessions.row_public_ids == ["1::0", "2::0", "2::1"]
        assert test_sessions.target_public_ids == other_test_dataloader.dataset.rows.target_public_ids

        # Every target comes right after the session items preceding it
        session_items = {"1::0": ["4", "5"], "2::0": ["2", "4", "6"], "2::1": ["2", "4", "6"]}

        _, lens = test_sessions.get_eval_context(range(test_sessions.n_rows), max_seq_len=3)
        for row, (target,), length in zip(
            test_sessions.row_public_ids, test_sessions.target_public_ids, lens.tolist()
        ):
            assert length == session_items[row].index(target)

    @pytest.mark.parametrize("models,expected_test_owners,expected_test_targets", [
        # FLAT models only: user 1's test session [1] is a FLAT row like any other
        ({"ItemKNN": {"neighborhood": 50}}, ["1", "2"], [["1"], ["6"]]),
        # FLAT and SESSION_ONLY models: user 1's single-item test session has no session
        # prefix, hence no SESSION_ONLY row, so its FLAT row is dropped as well
        ({"ItemKNN": {"neighborhood": 50}, "SASRec": {"meta": {"session_strategy": "session_only"}}},
         ["2"], [["6"]]),
    ])
    def test_flat_rows_aligned_with_sessions(self, models, expected_test_owners, expected_test_targets):
        config = {
            "models": models,
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    "leave_n_out": 1
                },
                "validation_splitting": {
                    "strategy": "temporal_holdout",
                    "leave_n_out": 1
                }
            }
        }

        val_data, main_data = _load_data(config_dict=config, load_as_session_only=True)

        test_rows = main_data.protocol.get_eval_dataloader(batch_size=10).dataset.rows
        assert test_rows.owner_public_ids == expected_test_owners
        assert test_rows.target_public_ids == expected_test_targets

        # Validation sessions [4, 5] and [2, 4, 6] have a prefix before their last item
        val_rows = val_data.protocol.get_eval_dataloader(batch_size=10).dataset.rows
        assert val_rows.owner_public_ids == ["1", "2"]
        assert val_rows.target_public_ids == [["5"], ["6"]]

    def test_mixed_session_strategies_share_rows(self):
        config = {
            "models": {
                "ItemKNN": {"neighborhood": 50},
                "SASRec": {"meta": {"session_strategy": "session_only"}}
            },
            "splitting": {
                "test_splitting": {
                    "strategy": "temporal_holdout",
                    "leave_n_out": 1
                },
                "validation_splitting": {
                    "strategy": "temporal_holdout",
                    "leave_n_out": 1
                }
            }
        }

        val_data, main_data = _load_data(config_dict=config, load_as_session_only=True)

        def check_same_rows(data):
            flat_rows = data.protocol.get_eval_dataloader(batch_size=10).dataset.rows
            session_rows = data.protocol.get_eval_dataloader(
                batch_size=10, session_strategy=SessionStrategy.SESSION_ONLY
            ).dataset.rows

            # Same owners and targets, whatever the context each strategy sees:
            # the whole history for FLAT, the session prefix for SESSION_ONLY
            assert flat_rows.owner_public_ids == session_rows.owner_public_ids
            assert flat_rows.target_public_ids == session_rows.target_public_ids

            _, flat_lens = flat_rows.get_eval_context(range(flat_rows.n_rows), max_seq_len=20)
            _, session_lens = session_rows.get_eval_context(range(session_rows.n_rows), max_seq_len=20)
            assert (flat_lens > session_lens).all()

        check_same_rows(main_data)
        check_same_rows(val_data)

    def test_negatives_shared_across_session_strategies_of_the_same_fold(self):
        # A single experiment can mix FLAT and SESSION_ONLY models on the same fold;
        # they must all be scored against the exact same negatives per user, so negative
        # sampling has to happen once, not once per requested strategy.
        num_negatives = 1

        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": num_negatives,
            }
        }

        # Load once, with sessions available, then request both strategies off the
        # very same fold's `EvalProtocol`, exactly like two different models would in one run.
        _, main_data = _load_data(config, load_as_session_only=True)

        flat_dataloader = main_data.protocol.get_eval_dataloader(batch_size=10)
        session_dataloader = main_data.protocol.get_eval_dataloader(
            batch_size=10, session_strategy=SessionStrategy.SESSION_ONLY
        )

        def negatives_by_row(dataloader):
            result = {}
            for users, eval_items in dataloader:
                for row, eval_ in zip(users.tolist(), eval_items):
                    result[row] = set(eval_[eval_ != -1].tolist())
            return result

        flat_negatives = negatives_by_row(flat_dataloader)
        session_negatives = negatives_by_row(session_dataloader)

        # User 2 (private 1) owns test rows 1 and 2 (its two sessions, per
        # `test_session_only_neg_random`); its candidate negative pool there is
        # {0, 2, 4}, disjoint from its ground-truth positives {1, 3, 5}, so those
        # positives can simply be filtered out to isolate the sampled negatives.
        positives = {1, 3, 5}

        row1_negatives = session_negatives[1] - positives
        row2_negatives = session_negatives[2] - positives
        flat_negatives_user2 = flat_negatives[1] - positives

        assert len(row1_negatives) == num_negatives
        # Sampled once per user and broadcast: both of user 2's session rows,
        # and its FLAT row, land on the exact same sampled negatives.
        assert row1_negatives == row2_negatives == flat_negatives_user2


class TestEvalDataloaderFailures:

    @pytest.mark.parametrize("params", p["invalid_session_only_target"])
    def test_session_only_unavailable_targets(self, params):
        with pytest.raises(ValueError):
            eval_data(params, load_as_session_only=True, session_strategy=SessionStrategy.SESSION_ONLY, batch_size=2)

    def test_session_only_requested_on_flat_dataset(self):
        with pytest.raises(ValueError):
            eval_data(session_strategy=SessionStrategy.SESSION_ONLY, batch_size=2)


if __name__ == "__main__":
    pytest.main()
