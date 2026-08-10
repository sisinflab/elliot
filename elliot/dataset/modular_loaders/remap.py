"""Public (shared/global) payload -> per-fold private-id view. Given a payload keyed by
the public ids a loader's `discover()`/`filter()` narrowed to, and a fold's own
`u_map`/`i_map`, builds a new payload reindexed by that fold's private ids `0..n-1`.
Called only by `Interactions.get_side_info()`, and must never mutate the shared payload
in place.
"""

from typing import Any, Dict, List

from elliot.dataset.modular_loaders.formats import EmbeddingPayload, TextPayload


def remap_embedding_payload(payload: EmbeddingPayload, inv_mapping: List[Any]) -> EmbeddingPayload:
    """Rebuild `payload` (public-id-keyed) into a new `EmbeddingPayload` indexed by
    private ids `0..len(private_order)-1`, where `private_order[private_id]` is that
    private id's public counterpart.

    A `row_loader`-backed (LAZY/MMAP) payload is only wrapped -- the private row is
    translated to the original public one and handed to the *same* underlying
    `row_loader`, so no heavy data is duplicated. A `dense`/`sparse` (MEMORY) payload
    is gathered into a fresh, fold-sized copy: the one unavoidable cost of handing
    back something indexable by private id with zero further lookups.

    A square pairwise payload (item-item/user-user similarity, where `col_ids` is the
    same domain as `row_ids` -- see `pairwise_raw_to_embedding_payload`) has both axes
    remapped together; any other `col_ids` (a feature/vocabulary axis, not a user/item
    id) is left untouched.

    Raises `KeyError` if any id in `private_order` is missing from `payload.id_map`:
    that would mean this fold's users/items aren't actually covered by the loader's
    domain, which `discover()`/`filter()`'s cross-loader intersection is supposed to
    already guarantee -- surfacing it loudly here is cheaper than a silent
    user/item misalignment downstream.
    """
    if payload.id_map is None:
        raise ValueError("Cannot remap an EmbeddingPayload with no id_map.")

    missing = [k for k in inv_mapping if k not in payload.id_map]
    if missing:
        raise KeyError(
            f"{len(missing)} id(s) from this fold are not covered by the loader's "
            f"payload domain (e.g. {missing[:5]}) -- discover()/filter() should "
            f"already guarantee every fold's users/items are covered; this signals a "
            f"domain-intersection bug."
        )

    public_rows = [payload.id_map[pub] for pub in inv_mapping]
    n = len(inv_mapping)
    private_row_ids = list(range(n))
    private_id_map = dict(zip(private_row_ids, private_row_ids))
    square = payload.col_ids is not None and list(payload.col_ids) == list(payload.row_ids or [])
    private_col_ids = private_row_ids if square else payload.col_ids

    if payload.row_loader is not None:
        original_loader = payload.row_loader

        def wrapped_row_loader(private_row, _rows=public_rows, _loader=original_loader):
            return _loader(_rows[private_row])

        shape = (n,) + tuple(payload.shape[1:]) if payload.shape else None
        return EmbeddingPayload(
            row_loader=wrapped_row_loader,
            row_ids=private_row_ids,
            id_map=private_id_map,
            col_ids=private_col_ids,
            shape=shape,
        )

    if payload.dense is not None:
        dense = payload.dense[public_rows]
        if square:
            dense = dense[:, public_rows]
        return EmbeddingPayload(
            dense=dense,
            row_ids=private_row_ids,
            id_map=private_id_map,
            col_ids=private_col_ids,
            shape=dense.shape
        )

    if payload.sparse is not None:
        sparse = payload.sparse.tocsr()[public_rows]
        if square:
            sparse = sparse.tocsc()[:, public_rows].tocsr()
        return EmbeddingPayload(
            sparse=sparse,
            row_ids=private_row_ids,
            id_map=private_id_map,
            col_ids=private_col_ids,
            shape=sparse.shape
        )

    raise ValueError("EmbeddingPayload has no dense/sparse/row_loader data to remap.")


def remap_text_payload(payload: TextPayload, inv_mapping: List[Any]) -> TextPayload:
    """`TextPayload` counterpart of `remap_embedding_payload`, same private-id
    contract (including the loud `KeyError` on an uncovered fold id) -- there's no
    row_loader-style variant to preserve since `TextPayload` is always plain dicts.
    """
    if payload.id_map is None:
        raise ValueError("Cannot remap a TextPayload with no id_map.")

    missing = [k for k in inv_mapping if k not in payload.id_map]
    if missing:
        raise KeyError(
            f"{len(missing)} id(s) from this fold are not covered by the loader's "
            f"payload domain (e.g. {missing[:5]})."
        )

    tokens = (
        {i: payload.tokens.get(pub, []) for i, pub in enumerate(inv_mapping)}
        if payload.tokens is not None else None
    )
    raw_text = (
        {i: payload.raw_text.get(pub) for i, pub in enumerate(inv_mapping)}
        if payload.raw_text is not None else None
    )
    private_id_map = {i: i for i in range(len(inv_mapping))}
    return TextPayload(
        tokens=tokens,
        raw_text=raw_text,
        id_map=private_id_map,
        vocab_size=payload.vocab_size
    )


def remap_pair_payload(
    payload: EmbeddingPayload, u_map: Dict[Any, int], i_map: Dict[Any, int]
) -> EmbeddingPayload:
    """Re-key a `(user, item)`-pair-identified `EmbeddingPayload` (e.g.
    `InteractionsTextualAttributes`) from public `(user, item)` pairs to this fold's
    private ones.

    Unlike `remap_embedding_payload`, this never raises on a missing id and never
    reindexes `0..n-1`: not every private `(user, item)` pair is expected to have a
    row (this data is inherently sparse, one row per *interaction* rather than per
    entity), so a pair whose user or item falls outside this fold is simply dropped.
    The underlying `dense`/`sparse`/`row_loader` is reused as-is (only the `id_map`
    keys change), since row positions themselves are untouched.
    """
    if payload.id_map is None:
        raise ValueError("Cannot remap an EmbeddingPayload with no id_map.")

    private_id_map = {
        (u_map[u], i_map[i]): row
        for (u, i), row in payload.id_map.items()
        if u in u_map and i in i_map
    }
    return EmbeddingPayload(
        dense=payload.dense,
        sparse=payload.sparse,
        row_loader=payload.row_loader,
        row_ids=list(private_id_map.keys()),
        id_map=private_id_map,
        col_ids=payload.col_ids,
        shape=payload.shape,
    )
