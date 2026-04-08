import torch
import numpy as np
from tqdm import tqdm


def get_recommendations(model, dataloader, dataset, k=10):
    preds_dict = {}

    iter_data = tqdm(
        dataloader,
        desc="Collecting",
        total=len(dataloader),
        leave=False
    )

    for users, eval_items in iter_data:
        recs = model.predict(users, eval_items)

        if eval_items is not None:
            mask = eval_items == -1
        else:
            eval_batch = dataset.train_set.sparse_ratings[users.tolist()]
            mask = eval_batch.nonzero()

        v, i = _get_top_k(recs, k, mask, eval_items)
        recs_dict = _get_recs_dict(v, i, users, dataset.get_inverse_mappings())

        preds_dict.update(recs_dict)

    return preds_dict


def _get_top_k(recs, k, mask, item_indices=None):
    device = recs.device

    if item_indices is not None and item_indices.device != device:
        item_indices = item_indices.to(device)

    if isinstance(mask, np.ndarray):
        mask = torch.as_tensor(mask, device=device)
    elif isinstance(mask, torch.Tensor) and mask.device != device:
        mask = mask.to(device)

    recs[mask] = -torch.inf

    k = min(k, recs.shape[1])
    v, i = torch.topk(recs, k=k, sorted=True)

    if item_indices is not None:
        i = item_indices.gather(1, i)

    return v.detach().cpu().numpy(), i.detach().cpu().numpy()


def _get_recs_dict(values, item_indices, user_indices, inverse_mappings):
    if not item_indices.size:
        return {}
    pr_users, pr_items = inverse_mappings
    mapped_items = np.array(pr_items)[item_indices]
    mat = [[*zip(item, val)] for item, val in zip(mapped_items, values)]
    proc_batch = dict(zip([pr_users[u_i] for u_i in user_indices], mat))
    return proc_batch
