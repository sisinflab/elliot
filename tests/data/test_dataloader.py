import pytest
import numpy as np
from collections import Counter, defaultdict

from elliot.dataset import DataSetLoader
from elliot.namespace import build_namespace
from elliot.utils.folder import path_joiner

from tests.params import data_folder, dataset_path

current_path = path_joiner(__file__)


def train_data(train_set=None, side_info=None, **kwargs):
    data_obj = (
        side_info 
        if side_info is not None else train_set 
        if train_set is not None else _load_data()
    )
    train_dataloader = data_obj.get_dataloader(**kwargs)
    return train_dataloader, data_obj

def _load_data():
    # User 1 rated every item, so it has no valid negative
    config_data = {
        "experiment": {
            "dataset": "dataloader",
            "data_config": {
                "strategy": "fixed",
                "data_folder": data_folder,
                "reader": {"header": True},
            },
        }
    }

    config = build_namespace(config_path=current_path, config_data=config_data)

    loader = DataSetLoader(config=config)
    _, main_data = loader.build()

    return main_data[0].train_set

def _load_kg_data():
    kg_path = path_joiner(data_folder, "kg_triples")
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
                "side_information": {
                    "dataloader": "KGTriplesLoader",
                    "kg_train": path_joiner(kg_path, "kg_train_direct.txt"),
                    "item_mapping": path_joiner(kg_path, "item_mapping.tsv"),
                },
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

    loader = DataSetLoader(config=config)
    _, main_data = loader.build()

    return main_data[0].side_info

def _usable_users(ratings, n_items):
    # Users a negative can be sampled for: at least one unrated item
    return {u for u, items in ratings.items() if len(items) < n_items}


class TestTrainDataloader:

    N_EVENTS = 200

    @pytest.mark.parametrize("sampler_name,params", [
        ("PairWiseSampler", {}),
        ("PairWiseBatchSampler", {"b_size": 4}),
    ])
    def test_pairwise(self, sampler_name, params):
        config = {
            "sampler_name": sampler_name,
            "transactions": self.N_EVENTS,
            "batch_size": 16,
            "seed": 0,
            **params
        }

        train_dataloader, train_set = train_data(**config)
        ratings = train_set.get_dict(private=True)
        usable = _usable_users(ratings, train_set.dims[1])

        total_rows = 0
        for user, pos, neg in train_dataloader:
            for u, i, j in zip(user.tolist(), pos.tolist(), neg.tolist()):
                # A user who rated every item is always redrawn
                assert u in usable
                assert i in ratings[u]
                assert j not in ratings[u]
            total_rows += len(user)

        assert total_rows == self.N_EVENTS

    def test_mf_pairwise(self):
        m = 3
        config = {
            "sampler_name": "MFPairWiseSampler",
            "m": m,
            "batch_size": 4,
            "seed": 0
        }

        train_dataloader, train_set = train_data(**config)
        ratings = train_set.get_dict(private=True)
        usable = _usable_users(ratings, train_set.dims[1])

        pairs = Counter()
        for user, pos, neg in train_dataloader:
            for u, i, j in zip(user.tolist(), pos.tolist(), neg.tolist()):
                assert i in ratings[u]
                assert j not in ratings[u]
                pairs[(u, i)] += 1

        # Exactly m triples per pair of every user with a valid negative
        assert pairs == {(u, i): m for u in usable for i in ratings[u]}

    @pytest.mark.parametrize("sampler_name,params,implicit", [
        ("PointWisePosNegSampler", {}, True),
        ("PointWisePosNegRatingsSampler", {}, False),
        ("PointWisePosNegRatioRatingsSampler", {"neg_ratio": 3}, False),
    ])
    def test_pointwise(self, sampler_name, params, implicit):
        config = {
            "sampler_name": sampler_name,
            "transactions": self.N_EVENTS,
            "batch_size": 16,
            "seed": 0,
            **params
        }

        train_dataloader, train_set = train_data(**config)
        ratings = train_set.get_dict(private=True)
        usable = _usable_users(ratings, train_set.dims[1])

        total_rows = 0
        for user, item, rating in train_dataloader:
            # Implicit labels are ints, explicit ratings floats (never truncated)
            assert rating.is_floating_point() != implicit
            for u, i, r in zip(user.tolist(), item.tolist(), rating.tolist()):
                assert u in usable
                if i in ratings[u]:
                    assert r == (1 if implicit else ratings[u][i])
                else:
                    assert r == 0
            total_rows += len(user)

        assert total_rows == self.N_EVENTS

    def test_mf_pointwise(self):
        m = 3
        config = {
            "sampler_name": "MFPointWisePosNegSampler",
            "m": m,
            "batch_size": 4,
            "seed": 0
        }

        train_dataloader, train_set = train_data(**config)
        ratings = train_set.get_dict(private=True)
        usable = _usable_users(ratings, train_set.dims[1])

        positives = Counter()
        total_rows = 0
        for user, item, label in train_dataloader:
            for u, i, l in zip(user.tolist(), item.tolist(), label.tolist()):
                # Positives are rated items, negatives never are
                assert (i in ratings[u]) == (l == 1)
                if l == 1:
                    positives[(u, i)] += 1
            total_rows += len(user)

        # One positive plus exactly m negatives per pair of every user with a valid negative
        assert positives == {(u, i): 1 for u in usable for i in ratings[u]}
        assert total_rows == len(positives) * (m + 1)

    def test_custom_pointwise_sparse(self):
        config = {
            "sampler_name": "CustomPointWiseSparseSampler",
            "transactions": self.N_EVENTS,
            "batch_size": 16,
            "seed": 0
        }

        train_dataloader, train_set = train_data(**config)
        ratings = train_set.get_dict(private=True)
        usable = _usable_users(ratings, train_set.dims[1])

        total_rows = 0
        for user, item, rating in train_dataloader:
            assert rating.is_floating_point()
            for u, i, r in zip(user.tolist(), item.tolist(), rating.tolist()):
                assert u in usable
                assert r == ratings[u][i]
            total_rows += len(user)

        assert total_rows == self.N_EVENTS

    def test_sparse(self):
        train_set = _load_data()
        config = {
            "sampler_name": "SparseSampler",
            "sparse": train_set.sparse_ratings,
            "batch_size": 3,
            "seed": 0
        }

        train_dataloader, _ = train_data(train_set=train_set, **config)
        matrix = train_set.sparse_ratings.toarray()

        rows = [tuple(row) for batch in train_dataloader for row in batch.tolist()]

        # Every row of the ratings matrix, once
        assert sorted(rows) == sorted(tuple(row) for row in matrix.tolist())

    def test_kg_triples(self):
        side_info = _load_kg_data()
        payload = side_info.get("KGTriplesLoader")["kg_triples"]
        heads, relations, tails = (
            np.asarray(x).tolist() for x in (payload.heads, payload.relations, payload.tails)
        )
        config = {
            "sampler_name": "KGTriplesSampler",
            "kg_heads": heads,
            "kg_relations": relations,
            "kg_tails": tails,
            "n_entities": payload.n_entities,
            "batch_size": 16,
            "seed": 0
        }

        train_dataloader, _ = train_data(side_info=side_info, **config)

        triples = set(zip(heads, relations, tails))
        observed_tails = defaultdict(set)
        for h, r, t in triples:
            observed_tails[(h, r)].add(t)

        total_rows = 0
        for head, relation, pos_tail, neg_tail in train_dataloader:
            for h, r, t, n in zip(head.tolist(), relation.tolist(), pos_tail.tolist(), neg_tail.tolist()):
                assert (h, r, t) in triples
                # The negative tail is never an observed tail of its (head, relation)
                assert 0 <= n < payload.n_entities
                assert n not in observed_tails[(h, r)]
            total_rows += len(head)

        assert total_rows == len(heads)


if __name__ == "__main__":
    pytest.main()
