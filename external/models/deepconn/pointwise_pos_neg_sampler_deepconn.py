"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import random
import numpy as np
import tensorflow as tf


class Sampler:
    def __init__(self, ui_dict, private_users, private_items, train_reviews_tokens, epochs, u_kernel_size,
                 i_kernel_size, pad_index):
        np.random.seed(42)
        random.seed(42)
        self._ui_dict = ui_dict
        self._users = list(self._ui_dict.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._ui_dict.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(ui_dict[u])) for u in ui_dict}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._train_reviews_tokens = train_reviews_tokens
        self._private_users = private_users
        self._private_items = private_items
        self._epochs = epochs
        self._u_kernel_size = u_kernel_size
        self._i_kernel_size = i_kernel_size
        self._pad_index = pad_index

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        actual_inter = (events // batch_size) * batch_size * self._epochs

        counter_inter = 1

        def sample():
            # sample user/item pair
            u = r_int(n_users)
            u_pos = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                sample()
            b = random.getrandbits(1)
            if b:
                i = u_pos[r_int(lui)]
            else:
                i = r_int(n_items)
                while i in u_pos:
                    i = r_int(n_items)

            # get user review and item reviews
            u_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['USER_ID'] == self._private_users[u]][
                    'tokens_position'].tolist()
            # u_review_tokens = [sublist + ([self._pad_index] * (self._u_kernel_size - len(sublist))) if len(
            #     sublist) < self._u_kernel_size else sublist for sublist in u_review_tokens]
            # u_review_tokens = [int(item) for sublist in u_review_tokens for item in sublist]
            i_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['ITEM_ID'] == self._private_items[i]][
                    'tokens_position'].tolist()
            # i_review_tokens = [sublist + ([self._pad_index] * (self._i_kernel_size - len(sublist))) if len(
            #     sublist) < self._i_kernel_size else sublist for sublist in i_review_tokens]
            # i_review_tokens = [int(item) for sublist in i_review_tokens for item in sublist]

            return u, i, float(b), u_review_tokens, i_review_tokens

        for ep in range(self._epochs):
            for _ in range(events):
                yield sample()
                if counter_inter == actual_inter:
                    return
                else:
                    counter_inter += 1

    def pipeline(self, events, batch_size):
        data = tf.data.Dataset.from_generator(
            generator=self.step,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
            args=(events, batch_size)
        )
        data = data.map(lambda a, b, c, d, e: (a, b, c, tf.expand_dims(d, 0), tf.expand_dims(e, 0)),
                        num_parallel_calls=tf.data.AUTOTUNE)
        data = data.map(
            lambda a, b, c, d, e: (a, b, c, tf.RaggedTensor.from_tensor(d), tf.RaggedTensor.from_tensor(e)),
            num_parallel_calls=tf.data.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
        data = data.map(lambda a, b, c, d, e: (a, b, c, tf.squeeze(d, 1), tf.squeeze(e, 1)),
                        num_parallel_calls=tf.data.AUTOTUNE)

        return data

    def step_eval(self, user):
        n_items = self._nitems

        def sample(u, i):
            u_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['USER_ID'] == self._private_users[u]][
                    'tokens_position'].tolist()
            u_review_tokens = [sublist + ([self._pad_index] * (self._u_kernel_size - len(sublist))) if len(
                sublist) < self._u_kernel_size else sublist for sublist in u_review_tokens]
            u_review_tokens = [int(it) for sublist in u_review_tokens for it in sublist]
            i_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['ITEM_ID'] == self._private_items[i]][
                    'tokens_position'].tolist()
            i_review_tokens = [sublist + ([self._pad_index] * (self._i_kernel_size - len(sublist))) if len(
                sublist) < self._i_kernel_size else sublist for sublist in i_review_tokens]
            i_review_tokens = [int(it) for sublist in i_review_tokens for it in sublist]

            return u, i, 1.0, u_review_tokens, i_review_tokens

        for item in range(n_items):
            yield sample(user, item)

    def pipeline_eval(self, user, batch_size):
        data = tf.data.Dataset.from_generator(
            generator=self.step_eval,
            output_signature=(
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.int32),
                tf.TensorSpec(shape=(), dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
                tf.TensorSpec(shape=(None,), dtype=tf.int32),
            ),
            args=(user,)
        )
        data = data.map(lambda a, b, c, d, e: (a, b, c, tf.expand_dims(d, 0), tf.expand_dims(e, 0)),
                        num_parallel_calls=tf.data.AUTOTUNE)
        data = data.map(
            lambda a, b, c, d, e: (a, b, c, tf.RaggedTensor.from_tensor(d), tf.RaggedTensor.from_tensor(e)),
            num_parallel_calls=tf.data.AUTOTUNE)
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.AUTOTUNE)
        data = data.map(lambda a, b, c, d, e: (a, b, c, tf.squeeze(d, 1), tf.squeeze(e, 1)),
                        num_parallel_calls=tf.data.AUTOTUNE)

        return data
