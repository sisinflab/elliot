import numpy as np
import torch
import random

np.random.seed(1234)
random.seed(1234)


def sampling(current_batch_size, ps, item_num, neg_ratio, sampling_sift_pos):
    neg_candidates = np.arange(item_num)

    if sampling_sift_pos:
        neg_items = []
        for probs in ps:
            u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)
            
            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(neg_candidates, (current_batch_size, neg_ratio), replace=True)

    neg_items = torch.from_numpy(neg_items)

    return neg_items
