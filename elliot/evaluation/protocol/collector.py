from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from elliot.evaluation.protocol.protocol import EvalProtocol


@dataclass
class CollectedRecs:
    """Everything `Evaluator.eval()` needs about the eval rows, keyed by public row id.

    Attributes:
        recs (Dict[Any, List[Tuple[Any, float]]]): Top-k (item, score) list per row.
        ground_truth (Dict[Any, Dict[Any, float]]): {target item: rating} per row.
        owner_map (Dict[Any, Any], optional): Owning user per row, when a user can own
            several rows (their metrics are averaged per user); None otherwise.
    """
    recs: Dict[Any, List[Tuple[Any, float]]]
    ground_truth: Dict[Any, Dict[Any, float]]
    owner_map: Optional[Dict[Any, Any]] = None


def collect(
    model,
    dataloader: DataLoader,
    protocol: EvalProtocol,
    k: int = 10
) -> CollectedRecs:
    """Score the eval rows of `dataloader` with `model` and keep each row's top-k items.

    Args:
        model: The recommender to evaluate.
        dataloader (DataLoader): The eval dataloader, from `EvalProtocol.get_eval_dataloader()`;
            only its rows (`dataloader.dataset.rows`) are scored.
        protocol (EvalProtocol): The fold's evaluation protocol.
        k (int): Number of recommendations kept per row. Defaults to 10.

    Returns:
        CollectedRecs: The recommendations, ground truth and row owners of the
            eval rows.
    """
    from elliot.recommender.base_recommender import SequentialRecommender

    preds_dict = {}

    # The dataloader iterates over its own eval rows: only their owners are scored
    eval_rows = dataloader.dataset.rows
    mask_seen = protocol.mask_seen
    inverse_mappings = protocol.dataset.get_inverse_mappings()

    iter_data = tqdm(
        dataloader,
        desc="Collecting",
        total=len(dataloader),
        leave=False
    )

    for rows, eval_items in iter_data:
        # `rows` holds eval-row indices: models only ever see each row's owning user
        users = torch.as_tensor(eval_rows.owner_users[rows.tolist()], dtype=torch.long)
        predict_kwargs = {"user_indices": users}

        if eval_items is not None:
            predict_kwargs["item_indices"] = eval_items

        if isinstance(model, SequentialRecommender):
            # The model never holds eval-only data: its input sequence is
            # built here from the eval rows and handed in
            user_seq, seq_len = eval_rows.get_eval_context(rows, model.max_seq_len)
            predict_kwargs["user_seq"] = user_seq
            predict_kwargs["seq_len"] = seq_len
            recs = model.predict(**predict_kwargs)
        elif eval_items is None:
            # Static models score a user the same on every row it owns
            unique_users, inverse = torch.unique(users, return_inverse=True)
            recs = model.predict(user_indices=unique_users)
            recs = recs[inverse.to(recs.device)]
        else:
            recs = model.predict(**predict_kwargs)

        if eval_items is not None:
            mask = eval_items == -1
        elif mask_seen:
            mask = eval_rows.get_seen_mask(rows)
        else:
            mask = None

        v, i = _get_top_k(recs, k, mask, eval_items)
        recs_dict = _get_recs_dict(v, i, rows, inverse_mappings, eval_rows.row_public_ids)

        preds_dict.update(recs_dict)

    return CollectedRecs(
        recs=preds_dict,
        ground_truth=eval_rows.ground_truth,
        owner_map=eval_rows.owner_map if eval_rows.multi_row else None
    )


def _get_top_k(
    recs,
    k,
    mask,
    item_indices=None
):
    device = recs.device

    if item_indices is not None and item_indices.device != device:
        item_indices = item_indices.to(device)

    if isinstance(mask, np.ndarray):
        mask = torch.as_tensor(mask, device=device)
    elif isinstance(mask, torch.Tensor) and mask.device != device:
        mask = mask.to(device)

    if mask is not None:
        recs[mask] = -torch.inf

    k = min(k, recs.shape[1])
    v, i = torch.topk(recs, k=k, sorted=True)

    if item_indices is not None:
        i = item_indices.gather(1, i)

    return v.detach().cpu().numpy(), i.detach().cpu().numpy()


def _get_recs_dict(
    values,
    item_indices,
    row_indices,
    inverse_mappings,
    row_public_ids
):
    if not item_indices.size:
        return {}
    _, pr_items = inverse_mappings
    mapped_items = np.array(pr_items)[item_indices]
    # A row with fewer than k valid items gets masked slots (seen items, -1 candidate
    # padding mapped to the last item) in its top-k: drop them
    mat = [
        [(item, v) for item, v in zip(items, val) if v != -np.inf]
        for items, val in zip(mapped_items, values)
    ]
    row_keys = [row_public_ids[r] for r in row_indices]
    proc_batch = dict(zip(row_keys, mat))
    return proc_batch
