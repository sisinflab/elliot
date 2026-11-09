import pytest
import numpy as np
import torch
from types import SimpleNamespace
from scipy.sparse import csr_matrix

from elliot.recommender.base_recommender import SequentialRecommender
from elliot.evaluation.protocol import CollectedRecs, collect

# Catalogue (private ids 0, 1, 2) and the score every fake model gives each item
ITEMS = ["i1", "i2", "i3"]
SCORES = [0.1, 0.2, 0.9]

# u1 (private 0) has seen i1, u2 (private 1) has seen i2
SEEN = csr_matrix(([1.0, 1.0], ([0, 1], [0, 1])), shape=(2, 3))

# (owner users, row public ids): one row per user, or u1 owning two rows
SINGLE_ROW = ([0, 1], ["u1", "u2"])
MULTI_ROW = ([0, 0, 1], ["u1::0", "u1::1", "u2::0"])


class _FakeRows:
    """Eval rows, each owned by a user and masking its owner's seen items."""

    def __init__(self, owner_users, row_public_ids):
        self.owner_users = np.asarray(owner_users)
        self.row_public_ids = row_public_ids
        self.multi_row = len(set(owner_users)) < len(owner_users)
        self.ground_truth = {row: {"i3": 1.0} for row in row_public_ids}
        self.owner_map = {row: row.split("::")[0] for row in row_public_ids}
        self.context_requests = []

    def get_seen_mask(self, row_indices):
        return SEEN[self.owner_users[np.asarray(row_indices)]].nonzero()

    def get_eval_context(self, row_indices, max_seq_len):
        self.context_requests.append(row_indices.tolist())
        n = len(row_indices)
        return torch.zeros((n, 1), dtype=torch.long), torch.ones(n, dtype=torch.long)


class _FakeDataLoader:
    def __init__(self, rows, batches):
        self.dataset = SimpleNamespace(rows=rows)
        self._batches = batches

    def __iter__(self):
        return iter(self._batches)

    def __len__(self):
        return len(self._batches)


def _scores(n, item_indices=None):
    """Score the whole catalogue, or each candidate slot as its item. Padding slots
    score highest of all: only masking keeps them out of the top-k."""
    scores = torch.tensor(SCORES).repeat(n, 1)
    if item_indices is None:
        return scores
    return torch.where(item_indices >= 0, scores.gather(1, item_indices.clamp(min=0)), 5.0)


class _FakeModel:
    def __init__(self):
        self.calls = []

    def predict(self, user_indices, item_indices=None, **kwargs):
        self.calls.append(user_indices.tolist())
        return _scores(len(user_indices), item_indices)


class _FakeSequentialModel(SequentialRecommender):
    max_seq_len = 3

    def __init__(self):
        self.calls = []

    def train_step(self, batch, *args):
        pass

    def predict(self, user_seq, seq_len, user_indices=None, item_indices=None, **kwargs):
        self.calls.append(user_indices.tolist())
        return _scores(len(user_seq), item_indices)


def _collect(model, rows, batches, k=2, mask_seen=True):
    """Collect `model`'s recommendations over `rows`, fed in `batches` of
    (row indices, candidate items or None for full ranking)."""
    protocol = SimpleNamespace(
        mask_seen=mask_seen,
        dataset=SimpleNamespace(get_inverse_mappings=lambda: (["u1", "u2"], ITEMS))
    )
    return collect(model, _FakeDataLoader(rows, batches), protocol, k)


def _items(recs):
    return {row: [item for item, _ in row_recs] for row, row_recs in recs.items()}


class TestFullRanking:

    @pytest.mark.parametrize("rows,expected", [
        (SINGLE_ROW, {"u1": ["i3", "i2"], "u2": ["i3", "i1"]}),
        (MULTI_ROW, {"u1::0": ["i3", "i2"], "u1::1": ["i3", "i2"], "u2::0": ["i3", "i1"]}),
    ])
    def test_masks_the_seen_items_of_each_row_owner(self, rows, expected):
        batches = [(torch.arange(len(rows[0])), None)]

        recs = _collect(_FakeModel(), _FakeRows(*rows), batches).recs

        assert _items(recs) == expected

    def test_mask_seen_disabled_keeps_seen_items(self):
        batches = [(torch.tensor([0, 1]), None)]

        recs = _collect(_FakeModel(), _FakeRows(*SINGLE_ROW), batches, k=3, mask_seen=False).recs

        assert _items(recs)["u1"] == ["i3", "i2", "i1"]
        assert [score for _, score in recs["u1"]] == pytest.approx([0.9, 0.2, 0.1])

    def test_masked_items_never_fill_a_top_k_beyond_the_unseen_ones(self):
        batches = [(torch.tensor([0, 1]), None)]

        recs = _collect(_FakeModel(), _FakeRows(*SINGLE_ROW), batches, k=3).recs

        assert _items(recs) == {"u1": ["i3", "i2"], "u2": ["i3", "i1"]}

    def test_merges_batches(self):
        batches = [(torch.tensor([0]), None), (torch.tensor([1]), None)]

        recs = _collect(_FakeModel(), _FakeRows(*SINGLE_ROW), batches).recs

        assert set(recs) == {"u1", "u2"}

    def test_static_model_scores_each_owner_once(self):
        model = _FakeModel()
        batches = [(torch.tensor([0, 1, 2]), None)]

        recs = _collect(model, _FakeRows(*MULTI_ROW), batches).recs

        assert model.calls == [[0, 1]]
        assert set(recs) == {"u1::0", "u1::1", "u2::0"}

    def test_sequential_model_gets_each_row_context_and_owner(self):
        model = _FakeSequentialModel()
        rows = _FakeRows(*MULTI_ROW)

        _collect(model, rows, [(torch.tensor([0, 1, 2]), None)])

        assert rows.context_requests == [[0, 1, 2]]
        assert model.calls == [[0, 0, 1]]


class TestNegativeSampling:
    """Candidate slots are mapped back to their items, and -1 padding is masked."""

    def test_ranks_each_row_candidates(self):
        # u1's rows rank different candidates; padding would win if not masked
        eval_items = torch.tensor([[0, 2, -1], [1, 2, -1], [0, 1, 2]])
        batches = [(torch.tensor([0, 1, 2]), eval_items)]

        recs = _collect(_FakeModel(), _FakeRows(*MULTI_ROW), batches).recs

        assert _items(recs) == {"u1::0": ["i3", "i1"], "u1::1": ["i3", "i2"], "u2::0": ["i3", "i2"]}
        assert [score for _, score in recs["u1::0"]] == pytest.approx([0.9, 0.1])

    def test_padding_never_fills_a_top_k_beyond_the_candidates(self):
        # A single candidate, or none at all (every item already seen)
        eval_items = torch.tensor([[0, -1, -1], [-1, -1, -1]])
        batches = [(torch.tensor([0, 1]), eval_items)]

        recs = _collect(_FakeModel(), _FakeRows(*SINGLE_ROW), batches, k=3).recs

        assert _items(recs) == {"u1": ["i1"], "u2": []}


class TestCollectedRecs:

    def test_single_row_hands_over_ground_truth_without_owner_map(self):
        rows = _FakeRows(*SINGLE_ROW)

        collected = _collect(_FakeModel(), rows, [(torch.tensor([0, 1]), None)])

        assert isinstance(collected, CollectedRecs)
        assert collected.ground_truth is rows.ground_truth
        assert collected.owner_map is None

    def test_multi_row_hands_over_ground_truth_and_owner_map(self):
        rows = _FakeRows(*MULTI_ROW)

        collected = _collect(_FakeModel(), rows, [(torch.tensor([0, 1, 2]), None)])

        assert collected.ground_truth is rows.ground_truth
        assert collected.owner_map == {"u1::0": "u1", "u1::1": "u1", "u2::0": "u2"}


if __name__ == "__main__":
    pytest.main()
