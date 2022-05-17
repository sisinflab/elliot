import numpy as np
import torch
import random

np.random.seed(1234)
random.seed(1234)


def sampling(pos_train_data, item_num, neg_ratio, interacted_items, sampling_sift_pos):
    neg_candidates = np.arange(item_num)

    if sampling_sift_pos:
        neg_items = []
        for u in pos_train_data[0]:
            probs = np.ones(item_num)
            probs[interacted_items[u]] = 0
            probs /= np.sum(probs)

            # u_neg_items = np.random.choice(neg_candidates, size=neg_ratio, p=probs, replace=True).reshape(1, -1)
            u_neg_items = np.repeat(100, neg_ratio).reshape(1, -1)
            
            neg_items.append(u_neg_items)

        neg_items = np.concatenate(neg_items, axis=0)
    else:
        neg_items = np.random.choice(neg_candidates, (len(pos_train_data[0]), neg_ratio), replace=True)

    neg_items = torch.from_numpy(neg_items)

    return pos_train_data[0], pos_train_data[1], neg_items
