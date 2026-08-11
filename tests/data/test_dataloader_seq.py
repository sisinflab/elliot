import pytest
import torch

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.enums import SessionStrategy
from elliot.utils.folder import path_joiner

from tests.params import data_folder, dataset_path

current_path = path_joiner(__file__)


def train_data(config_dict={}, load_as_session_only=False, **kwargs):
    train_val_data, _ = _load_data(config_dict, load_as_session_only=load_as_session_only)
    train_dataloader = train_val_data.train_sessions.get_dataloader(**kwargs)
    return train_dataloader, train_val_data

def eval_data(config_dict={}, load_as_session_only=False, **kwargs):
    val_data, main_data = _load_data(config_dict, load_as_session_only=load_as_session_only)
    val_dataloader = val_data.get_eval_dataloader(**kwargs)
    test_dataloader = main_data.get_eval_dataloader(**kwargs)
    return val_dataloader, test_dataloader, val_data, main_data

def _load_data(config_dict={}, load_as_session_only=False):
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
                "simple_metrics": ["nDCG"]
            },
            **config_dict
        }
    }

    config = build_namespace(config_path=current_path, config_data=config_data)

    loader = DataSetLoader(config=config)
    val_data, main_data = loader.build()

    loader.prepare_dataset(val_data, main_data)

    return val_data[0][0], main_data[0]


class TestTrainDataloader:

    @pytest.mark.parametrize("strategy,expected", [
        (SessionStrategy.FLAT, {
            ((0,), 1, 1),
            ((0, 1), 2, 2),
            ((0, 1, 2), 3, 3),
            ((1, 2, 3), 3, 4),
            ((2, 3, 4), 3, 3),
            ((1,), 1, 3),
            ((1, 3), 2, 5),
            ((1, 3, 5), 3, 3),
            ((3, 5, 3), 3, 5),
            ((5, 3, 5), 3, 1),
        }),
        (SessionStrategy.SESSION_ONLY, {
            ((0,), 1, 1),
            ((0, 1), 2, 2),
            ((3,), 1, 4),
            ((3, 4), 2, 3),
            ((1,), 1, 3),
            ((1, 3), 2, 5),
            ((3,), 1, 5),
            ((3, 5), 2, 1),
        }),
    ])
    def test_sessions_sequential(self, strategy, expected):
        config = {
            "sampler_name": "SequentialSampler",
            "strategy": strategy,
            "batch_size": 3,
            "max_seq_len": 3,
            "neg_samples": 1,
            "seed": 0
        }

        train_dataloader, _ = train_data(load_as_session_only=True, **config)

        actual = set()
        total_rows = 0
        for seq, length, target, negs in train_dataloader:
            assert (negs.squeeze(-1) != target).all()
            for i in range(len(seq)):
                actual.add((tuple(seq[i, :length[i]].tolist()), int(length[i]), int(target[i])))
            total_rows += len(seq)

        assert total_rows == len(expected)
        assert actual == expected

    def test_sessions_same_target(self):
        config = {
            "sampler_name": "SameTargetSequentialSampler",
            "strategy": SessionStrategy.FLAT,
            "batch_size": 3,
            "max_seq_len": 3,
            "seed": 0
        }

        train_dataloader, _ = train_data(**config)

        total_rows = 0
        for seq, _, target, sem_seq, _, has_semantic in train_dataloader:
            assert seq.shape[1] == 3
            assert sem_seq.shape == seq.shape
            assert has_semantic.dtype == torch.bool

            # Items 1, 3 and 5 are each a valid FLAT target more than once, so
            # every row targeting one of them has a semantic-positive pair;
            # every other target (0, 2, 4) is a valid target only once.
            assert torch.equal(has_semantic, torch.isin(target, torch.tensor([1, 3, 5])))

            total_rows += len(seq)

        assert total_rows == 17

    @pytest.mark.parametrize("strategy,expected", [
        (SessionStrategy.FLAT, {
            (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 3), (1, 3, 5), (3, 5, 3), (5, 3, 5), (3, 5, 1),
        }),
        (SessionStrategy.SESSION_ONLY, {
            (0, 1, 2), (3, 4, 3), (1, 3, 5), (3, 5, 1),
        }),
    ])
    def test_sessions_sliding_window(self, strategy, expected):
        config = {
            "sampler_name": "SlidingWindowSampler",
            "strategy": strategy,
            "batch_size": 3,
            "max_seq_len": 3,
            "stride": 1,
            "seed": 0
        }

        train_dataloader, _ = train_data(load_as_session_only=True, **config)

        actual = set()
        total_rows = 0
        for (pos_seq,) in train_dataloader:
            actual.update(tuple(row.tolist()) for row in pos_seq)
            total_rows += len(pos_seq)

        assert total_rows == len(expected)
        assert actual == expected

    @pytest.mark.parametrize("strategy,expected", [
        (SessionStrategy.FLAT, {(3, 4, 3), (3, 5, 1)}),
        (SessionStrategy.SESSION_ONLY, {(0, 1, 2), (3, 4, 3), (1, 3, 5), (3, 5, 1)}),
    ])
    def test_sessions_cloze(self, strategy, expected):
        mask_token_id = 6
        config = {
            "sampler_name": "ClozeSampler",
            "strategy": strategy,
            "batch_size": 3,
            "max_seq_len": 3,
            "mask_prob": 0.5,
            "mask_token_id": mask_token_id,
            "neg_samples": 1,
            "seed": 0
        }

        train_dataloader, _ = train_data(load_as_session_only=True, **config)

        total_rows = 0
        for masked_seq, pos_items, neg_items, masked_idx in train_dataloader:
            assert pos_items.shape == masked_seq.shape
            assert neg_items.shape == (*masked_seq.shape, 1)
            assert masked_idx.shape == masked_seq.shape

            # Every window here is exactly `max_seq_len` long (no padding), and
            # mask_prob=0.5 of 3 positions rounds down to exactly one masked
            # slot, so `mask_token_id` can only appear at that masked position.
            # `pos_items`/`neg_items` are indexed by mask-slot order (slot 0 is
            # the only one filled here), not by sequence position; `masked_idx`
            # carries the actual sequence position for that slot.
            for row in range(len(masked_seq)):
                pos = int(masked_idx[row, 0])
                assert masked_seq[row, pos] == mask_token_id

                reconstructed = masked_seq[row].clone()
                reconstructed[pos] = pos_items[row, 0]
                assert tuple(reconstructed.tolist()) in expected

                assert neg_items[row, 0, 0] != pos_items[row, 0]

            total_rows += len(masked_seq)

        assert total_rows == len(expected)

    def test_session_only_requested_on_flat_dataset_falls_back(self):
        params = {
            "sampler_name": "SequentialSampler",
            "strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 2,
            "max_seq_len": 3,
            "neg_samples": 1,
            "seed": 0
        }

        train_dataloader, train_val_data = train_data(**params)

        assert train_val_data.train_sessions._has_sessions is False

        batches = list(train_dataloader)
        assert batches


class TestEvalDataloader:

    def test_session_only_full_eval_rows(self):
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 1
        }

        val_dataloader, test_dataloader, val_data, main_data = eval_data(load_as_session_only=True, **params)

        def check_seen_users(dataloader, expected_users):
            seen_users = []
            for users, eval_items in dataloader:
                assert eval_items is None
                seen_users.extend(users.tolist())
            assert sorted(seen_users) == expected_users

        check_seen_users(test_dataloader, [0, 1, 2, 3])
        check_seen_users(val_dataloader, [0, 1])

        test_sessions = main_data.eval_sessions
        val_sessions = val_data.eval_sessions

        owner = test_sessions.owner_users
        assert owner is not None
        assert list(owner) == [0, 0, 1, 1]

        row_public_ids = test_sessions.row_public_ids
        assert row_public_ids == ["1::s0", "1::s1", "2::s0", "2::s1"]

        # userId is read back as string by the loader's default dtype config,
        # so the owner map's values are the raw (string) public user ids.
        owner_map = test_sessions.owner_map
        assert owner_map == {"1::s0": "1", "1::s1": "1", "2::s0": "2", "2::s1": "2"}

        # Each row's ground truth is that session's own masked (last) item,
        # never shared with another session of the same owner: user 1's two
        # rows disagree (5 vs 1) even though both are owned by user 1.
        assert test_sessions.target_public_ids == ["5", "1", "6", "6"]

        # Validation has exactly 1 session/user, so unlike the test fold above
        # there's no owner with more than one row to disambiguate.
        owner = val_sessions.owner_users
        assert owner is not None
        assert list(owner) == [0, 1]

        row_public_ids = val_sessions.row_public_ids
        assert row_public_ids == ["1::s0", "2::s0"]

        owner_map = val_sessions.owner_map
        assert owner_map == {"1::s0": "1", "2::s0": "2"}

        assert val_sessions.target_public_ids == ["3", "6"]

    def test_session_only_context_is_leave_last_item_out(self):
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 10
        }

        _, _, val_data, main_data = eval_data(load_as_session_only=True, **params)

        test_sessions = main_data.eval_sessions
        val_sessions = val_data.eval_sessions

        _, lens = test_sessions.get_eval_context([0, 1, 2, 3], max_seq_len=3)

        # row 0: user 1's older test session [4, 5] -> leave-last-out context [4], len 1
        # row 1: user 1's newer test session [1]    -> leave-last-out context [],  len 0
        # rows 2/3: user 2's two test sessions [2, 4, 6] each -> context [2, 4], len 2
        assert lens.tolist() == [1, 0, 2, 2]

        _, val_lens = val_sessions.get_eval_context([0, 1], max_seq_len=3)

        # row 0: user 1's validation session [1, 2, 3] -> leave-last-out context [1, 2], len 2
        # row 1: user 2's validation session [2, 4, 6] -> leave-last-out context [2, 4], len 2
        assert val_lens.tolist() == [2, 2]

    @pytest.mark.parametrize("leave_one_out", [False, True])
    def test_session_only_neg_random(self, leave_one_out):
        num_negatives = 20
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": num_negatives,
                "leave_one_out": leave_one_out
            }
        }
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 10
        }

        val_dataloader, test_dataloader, _, _ = eval_data(config, load_as_session_only=True, **params)

        # Private ids per the module docstring: user 1 -> private 0 with test
        # rows 0 (session [4, 5], target 5 -> private 4) and 1 (session [1],
        # target 1 -> private 0); user 2 -> private 1 with rows 2 and 3, both
        # session [2, 4, 6], target 6 -> private 5.
        target = {0: 4, 1: 0, 2: 5, 3: 5}
        owner_of = {0: 0, 1: 0, 2: 1, 3: 1}

        # Allowed negatives: every item never seen by that owning user, in
        # train or in *any* of their eval sessions (not just this row's own
        # session). num_negatives=20 exceeds every candidate pool here, so
        # the sampler always returns the full allowed set, deterministically.
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
                    # row (that session's own masked/last item); leave_one_out
                    # can't reduce that further, so it has no observable
                    # effect here.
                    expected = allowed_negatives[owner_of[row]] | {target[row]}
                    assert set(eval_) == expected

            assert seen_rows == expected_rows

        check_dataloader(test_dataloader, {0, 1, 2, 3}, target, owner_of)

        # Validation: user 1's single session [1, 2, 3] -> target public "3"
        # (private 2); user 2's single session [2, 4, 6] -> target public "6"
        # (private 5). Both users' validation-session items are already
        # covered by train, so the allowed-negative pool per owner is
        # identical to the test-set case above.
        val_target = {0: 2, 1: 5}
        val_owner_of = {0: 0, 1: 1}

        check_dataloader(val_dataloader, {0, 1}, val_target, val_owner_of)

    def test_session_only_requested_on_flat_dataset_falls_back(self):
        params = {
            "session_strategy": SessionStrategy.SESSION_ONLY,
            "batch_size": 2
        }

        _, test_dataloader, _, main_data = eval_data(**params)

        assert main_data.session_only_evaluation is False
        assert main_data.eval_sessions is None

        batches = list(test_dataloader)
        assert batches

    def test_negatives_shared_across_session_strategies_of_the_same_fold(self):
        # A single experiment can mix FLAT and SESSION_ONLY models on the same fold
        # (each requesting its own `session_strategy` from the shared `DataSet`); they
        # must all be scored against the exact same negatives per user, so negative
        # sampling has to happen once, not once per requested strategy.
        num_negatives = 1  # Below the 3-item candidate pool of user 2 (private 1):
                            # forces genuine subsampling instead of the full pool being
                            # trivially returned every time.
        config = {
            "negative_sampling": {
                "strategy": "random",
                "num_negatives": num_negatives,
            }
        }

        # Load once, with sessions available, then request both strategies off the
        # very same `DataSet`, exactly like two different models would in one run.
        _, main_data = _load_data(config, load_as_session_only=True)

        flat_dataloader = main_data.get_eval_dataloader(batch_size=10)
        session_dataloader = main_data.get_eval_dataloader(
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

        # User 2 (private 1) owns test rows 2 and 3 (its two sessions, per
        # `test_session_only_neg_random`); its candidate negative pool there is
        # {0, 2, 4}, disjoint from its ground-truth positives {1, 3, 5}, so those
        # positives can simply be filtered out to isolate the sampled negatives.
        positives = {1, 3, 5}

        row2_negatives = session_negatives[2] - positives
        row3_negatives = session_negatives[3] - positives
        flat_negatives_user2 = flat_negatives[1] - positives

        assert len(row2_negatives) == num_negatives
        # Sampled once per user and broadcast: both of user 2's session rows,
        # and its FLAT row, land on the exact same sampled negatives.
        assert row2_negatives == row3_negatives == flat_negatives_user2


if __name__ == "__main__":
    pytest.main()
