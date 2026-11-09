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
                "simple_metrics": ["nDCG"],
                "protocol": ({"target": "last"} if load_as_session_only else {})
            },
            **config_dict
        }
    }

    config = build_namespace(config_path=current_path, config_data=config_data)

    loader = DataSetLoader(config=config)
    val_data, main_data = loader.build()

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

    @pytest.mark.parametrize("strategy,expected", [
        (SessionStrategy.FLAT, {
            (0, (0,), 1, 1),
            (0, (0, 1), 2, 2),
            (0, (0, 1, 2), 3, 3),
            (0, (1, 2, 3), 3, 4),
            (0, (2, 3, 4), 3, 3),
            (1, (1,), 1, 3),
            (1, (1, 3), 2, 5),
            (1, (1, 3, 5), 3, 3),
            (1, (3, 5, 3), 3, 5),
            (1, (5, 3, 5), 3, 1),
        }),
        (SessionStrategy.SESSION_ONLY, {
            (0, (0,), 1, 1),
            (0, (0, 1), 2, 2),
            (0, (3,), 1, 4),
            (0, (3, 4), 2, 3),
            (1, (1,), 1, 3),
            (1, (1, 3), 2, 5),
            (1, (3,), 1, 5),
            (1, (3, 5), 2, 1),
        }),
    ])
    def test_user_sessions_sequential(self, strategy, expected):
        config = {
            "sampler_name": "UserSequentialSampler",
            "strategy": strategy,
            "batch_size": 3,
            "max_seq_len": 3,
            "neg_samples": 1,
            "seed": 0
        }

        train_dataloader, _ = train_data(load_as_session_only=True, **config)

        actual = set()
        total_rows = 0
        for user, seq, length, target, negs in train_dataloader:
            assert (negs.squeeze(-1) != target).all()
            for i in range(len(seq)):
                actual.add((int(user[i]), tuple(seq[i, :length[i]].tolist()), int(length[i]), int(target[i])))
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
        (SessionStrategy.FLAT, {
            (0, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, 3), (1, 3, 5), (3, 5, 3), (5, 3, 5), (3, 5, 1),
        }),
        (SessionStrategy.SESSION_ONLY, {
            (0, 1, 2), (3, 4, 3), (1, 3, 5), (3, 5, 1),
        }),
    ])
    def test_sessions_cloze(self, strategy, expected):
        mask_token_id = 6
        config = {
            "sampler_name": "ClozeSampler",
            "strategy": strategy,
            "batch_size": 3,
            "max_seq_len": 3,
            "stride": 1,
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


if __name__ == "__main__":
    pytest.main()
