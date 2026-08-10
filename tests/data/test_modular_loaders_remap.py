import pytest
from types import SimpleNamespace
import numpy as np
import pandas as pd

from elliot.dataset import Interactions
from elliot.dataset.modular_loaders import AbstractLoader, SideInformation
from elliot.dataset.modular_loaders.formats import EmbeddingPayload
from elliot.utils.enums import EntityAxis


def _fake_ns(fields=None):
    fields = fields or {}
    ns = SimpleNamespace(reader=SimpleNamespace(sep="\t", header=False, encoding=None), **fields)
    ns.model_dump = lambda: fields
    return ns


class _FakeItemLoader(AbstractLoader):
    """Minimal `AbstractLoader` for exercising `Interactions.get_side_info()`'s
    automatic private-id remap in isolation, without a real raw source: `load()`
    just turns the current (global) `self.items` domain into a one-column
    `EmbeddingPayload` whose value is the public id itself, so a remapped row's
    value directly reveals which public id ended up there.
    """

    provides = "item_features"
    format = "embedding"
    entity_axis = {"item_features": EntityAxis.ITEM}

    def load(self):
        row_ids = sorted(self.items)
        id_map = {i: idx for idx, i in enumerate(row_ids)}
        dense = np.array([[float(i)] for i in row_ids])
        return {"item_features": EmbeddingPayload(dense=dense, row_ids=row_ids, id_map=id_map, shape=dense.shape)}


def _make_fold(items_public_order, side_info, user=100):
    """Build a minimal single-user `Interactions` whose fold-private item order is
    exactly `items_public_order` -- deliberately not required to be sorted/a
    superset of the loader's domain, to stress the general remap rather than the
    sorted-ascending order every real `DataSet` fold happens to produce.
    """
    df = pd.DataFrame({
        "userId": [user] * len(items_public_order),
        "itemId": items_public_order,
        "rating": [1.0] * len(items_public_order),
    })
    u_map = {user: 0}
    i_map = {item: idx for idx, item in enumerate(items_public_order)}
    inv_mappings = ([user], list(items_public_order))
    return Interactions(
        dataframe=df, name="train", u_map=u_map, i_map=i_map, inv_mappings=inv_mappings, side_info=side_info
    )


class TestInteractionsPrivateSideInfoView:
    """End-to-end through `Interactions.get_side_info()` (not just the adapter
    functions in `tests/data/test_modular_loaders_adapters.py` in isolation): the
    automatic remap, its per-fold caching, and the cross-fold cleanup
    `SideInformation.marked_as_done()` triggers.
    """

    def _loader(self, users, items, cls=_FakeItemLoader):
        return cls(users=users, items=items, ns=_fake_ns(), logger=None)

    def test_get_side_info_returns_view_in_this_folds_private_order(self):
        loader = self._loader(users={100}, items={10, 20, 30, 40})
        side_info = SideInformation({"FakeItemLoader": loader})

        # This fold's own train items are a reordered *subset* of the loader's global
        # domain -- exactly the case a naive "row index == private id" read would get wrong.
        fold = _make_fold([30, 10], side_info)

        payloads = fold.get_side_info("FakeItemLoader")
        assert payloads["item_features"].dense.tolist() == [[30.0], [10.0]]

        # Cached: a second call doesn't rebuild it.
        assert fold.get_side_info("FakeItemLoader") is payloads

    def test_two_folds_with_different_private_orders_each_get_their_own_correct_view(self):
        loader = self._loader(users={100}, items={10, 20, 30, 40})
        side_info = SideInformation({"FakeItemLoader": loader})

        fold_a = _make_fold([10, 20], side_info)
        fold_b = _make_fold([40, 20, 10], side_info)

        assert fold_a.get_side_info("FakeItemLoader")["item_features"].dense.tolist() == [[10.0], [20.0]]
        assert fold_b.get_side_info("FakeItemLoader")["item_features"].dense.tolist() == [[40.0], [20.0], [10.0]]

    def test_marked_as_done_drops_every_folds_cached_private_view(self):
        loader = self._loader(users={100}, items={10, 20, 30, 40})
        side_info = SideInformation({"FakeItemLoader": loader})

        fold_a = _make_fold([10, 20], side_info)
        fold_b = _make_fold([30, 40], side_info)
        fold_a.get_side_info("FakeItemLoader")
        fold_b.get_side_info("FakeItemLoader")
        assert fold_a.side_information.get("FakeItemLoader") is not None
        assert fold_b.side_information.get("FakeItemLoader") is not None

        side_info.mapped_uses([SimpleNamespace(_loaders=["FakeItemLoader"])])
        side_info.marked_as_done("FakeItemLoader")

        assert fold_a.side_information.get("FakeItemLoader") is None
        assert fold_b.side_information.get("FakeItemLoader") is None
        # The shared payload itself is gone too, so a further request rebuilds it.
        assert side_info._payloads["FakeItemLoader"] is None

    def test_axis_none_payload_is_returned_unremapped(self):
        """A loader that never opts a key into `entity_axis` (the default) gets that
        payload back exactly as `load()` produced it -- e.g. a `GraphPayload`, which
        this remap doesn't know how to reindex by a single user/item axis.
        """
        class _GraphLikeLoader(_FakeItemLoader):
            entity_axis = {}  # nothing declared -> EntityAxis.NONE for every key

            def load(self):
                return {"item_features": object()}

        loader = self._loader(users={100}, items={10, 20}, cls=_GraphLikeLoader)
        side_info = SideInformation({"GraphLike": loader})
        fold = _make_fold([10, 20], side_info)

        payloads = fold.get_side_info("GraphLike")
        shared = side_info.get_payload("GraphLike")
        assert payloads["item_features"] is shared["item_features"]


if __name__ == "__main__":
    pytest.main()
