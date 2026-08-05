import numpy as np
import pytest
import torch
from scipy.sparse import csr_matrix

from elliot.recommender.collector import get_recommendations


class _FakeTrainSet:
    def __init__(self, sparse):
        self.sparse_ratings = sparse


class _FakeEvalSessions:
    def __init__(self, owner_users, row_public_ids):
        self.owner_users = owner_users
        self.row_public_ids = row_public_ids


class _FakeDataset:
    """Classic (non session-only) fake dataset: eval rows == users."""

    def __init__(self):
        self.session_only_evaluation = False
        # user0 has seen item0 (i1), user1 has seen item1 (i2)
        sparse = csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(2, 3))
        self.train_set = _FakeTrainSet(sparse)

    def get_inverse_mappings(self):
        return (["u1", "u2"], ["i1", "i2", "i3"])


class _FakeSessionDataset(_FakeDataset):
    """SESSION_ONLY fake dataset: eval rows are per-session, owned by users."""

    def __init__(self):
        super().__init__()
        self.session_only_evaluation = True
        self.eval_sessions = _FakeEvalSessions(
            owner_users=np.array([0, 0, 1]),
            row_public_ids=["u1::s0", "u1::s1", "u2::s0"],
        )


class _FakeModel:
    def predict(self, user_indices, *args, item_indices=None, **kwargs):
        # i3 always scores highest, i1/i2 lower - only masking should ever exclude them
        n = len(user_indices)
        return torch.tensor([[0.1, 0.2, 0.9]] * n)


class _FakeDataLoader:
    def __init__(self, batches):
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


class TestFullEval:
    """eval_items is None: masking comes from each row's train history."""

    def test_keys_by_public_user_row_index_equals_user_index(self):
        dataset = _FakeDataset()
        model = _FakeModel()
        dataloader = _FakeDataLoader([(torch.tensor([0, 1]), None)])

        preds = get_recommendations(model, dataloader, dataset, k=2)

        # classic behaviour: keyed by public user id, row index == user index
        assert set(preds.keys()) == {"u1", "u2"}

        # user0 has seen i1 -> excluded; user1 has seen i2 -> excluded
        assert "i1" not in [item for item, _ in preds["u1"]]
        assert "i2" not in [item for item, _ in preds["u2"]]
    
    def test_session_only_masks_by_owning_user_not_row_index(self):
        dataset = _FakeSessionDataset()
        model = _FakeModel()
        dataloader = _FakeDataLoader([(torch.tensor([0, 1, 2]), None)])

        preds = get_recommendations(model, dataloader, dataset, k=2)

        assert set(preds.keys()) == {"u1::s0", "u1::s1", "u2::s0"}

        # rows 0 and 1 share owner u1 (has seen i1) -> i1 excluded from both
        for row_key in ("u1::s0", "u1::s1"):
            items = [item for item, _ in preds[row_key]]
            assert "i1" not in items
            assert "i3" in items

        # row 2's owner is u2 (has seen i2) -> i2 excluded
        items_row2 = [item for item, _ in preds["u2::s0"]]
        assert "i2" not in items_row2
        assert "i3" in items_row2

    def test_aggregates_predictions_across_multiple_batches(self):
        dataset = _FakeDataset()
        model = _FakeModel()
        dataloader = _FakeDataLoader([
            (torch.tensor([0]), None),
            (torch.tensor([1]), None),
        ])

        preds = get_recommendations(model, dataloader, dataset, k=2)

        assert set(preds.keys()) == {"u1", "u2"}


class _FakeNegSamplingModel:
    def predict(self, user_indices, *args, item_indices=None, **kwargs):
        # candidate slot 1 always scores highest, slot 0 next; padded slots
        # would score highest of all but must be masked out via eval_items == -1
        n = len(user_indices)
        return torch.tensor([[0.5, 0.9, 5.0]] * n)


class TestNegSamplingEval:
    """eval_items is not None: masking comes from -1 padding, regardless of
    session_only, while row keys still follow row_public_ids when present."""

    def test_masks_padding_and_keys_by_public_user(self):
        dataset = _FakeDataset()
        model = _FakeNegSamplingModel()
        # u1 candidates: i1 (idx0), i3 (idx2), padded; u2 candidates: i2 (idx1), i3 (idx2), padded
        eval_items = torch.tensor([[0, 2, -1], [1, 2, -1]])
        dataloader = _FakeDataLoader([(torch.tensor([0, 1]), eval_items)])

        preds = get_recommendations(model, dataloader, dataset, k=2)

        assert set(preds.keys()) == {"u1", "u2"}
        assert preds["u1"] == [("i3", pytest.approx(0.9)), ("i1", pytest.approx(0.5))]
        assert preds["u2"] == [("i3", pytest.approx(0.9)), ("i2", pytest.approx(0.5))]

    def test_session_only_masks_padding_but_keys_by_row_public_id(self):
        dataset = _FakeSessionDataset()
        model = _FakeNegSamplingModel()
        # row0/row1 (u1's sessions) candidates: i1, i3, padded
        # row2 (u2's session) candidates: i2, i3, padded
        eval_items = torch.tensor([[0, 2, -1], [0, 2, -1], [1, 2, -1]])
        dataloader = _FakeDataLoader([(torch.tensor([0, 1, 2]), eval_items)])

        preds = get_recommendations(model, dataloader, dataset, k=2)

        assert set(preds.keys()) == {"u1::s0", "u1::s1", "u2::s0"}
        assert preds["u1::s0"] == [("i3", pytest.approx(0.9)), ("i1", pytest.approx(0.5))]
        assert preds["u1::s1"] == [("i3", pytest.approx(0.9)), ("i1", pytest.approx(0.5))]
        assert preds["u2::s0"] == [("i3", pytest.approx(0.9)), ("i2", pytest.approx(0.5))]


if __name__ == "__main__":
    pytest.main()
