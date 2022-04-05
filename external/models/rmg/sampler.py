import numpy as np

np.random.seed(1234)


def generate_batch_data_random(item,
                               user,
                               user_to_item_to_user,
                               user_to_item,
                               item_to_user_to_item,
                               item_to_user,
                               item_id,
                               user_id,
                               y,
                               batch_size):
    idx = np.arange(user_id.shape[0])
    np.random.shuffle(idx)
    batches = [idx[range(batch_size * i, min(len(y), batch_size * (i + 1)))] for i in range(len(y) // batch_size + 1)]

    for i in batches:
        yield ([item[item_id[i]], user[user_id[i]], user_to_item_to_user[user_id[i]],
                user_to_item[user_id[i]], item_to_user_to_item[item_id[i]], item_to_user[item_id[i]],
                np.expand_dims(item_id[i], axis=1), np.expand_dims(user_id[i], axis=1)], y[i])
