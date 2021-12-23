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
    def __init__(self, ui_dict, iu_dict, private_users, private_items, train_reviews_tokens, epochs):
        np.random.seed(42)
        random.seed(42)
        self._ui_dict = ui_dict
        self._users = list(self._ui_dict.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._ui_dict.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(ui_dict[u])) for u in ui_dict}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._iu_dict = iu_dict
        self._train_reviews_tokens = train_reviews_tokens
        self._private_users = private_users
        self._private_items = private_items
        self._epochs = epochs

    def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        iu_dict = self._iu_dict
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

            i_pos = iu_dict[i]

            # get user ratings and item ratings
            u_ratings = np.zeros((1, self._nitems))
            i_ratings = np.zeros((1, self._nusers))
            u_ratings[0, u_pos] = 1.0
            i_ratings[0, i_pos] = 1.0

            # get user review and item review tokens
            u_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['USER_ID'] == self._private_users[u]][
                    'TOKENS_POSITION'].tolist()
            i_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['ITEM_ID'] == self._private_items[i]][
                    'TOKENS_POSITION'].tolist()

            return u, i, float(b), u_ratings, i_ratings, u_review_tokens, i_review_tokens

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
            output_types=(tf.int64, tf.int64, tf.float32, tf.float32, tf.float32, tf.int64, tf.int64),
            args=(events, batch_size)
        )
        data = data.map(lambda a, b, c, d, e, f, g: (a, b, c, d, e, tf.expand_dims(f, 0), tf.expand_dims(g, 0)))
        data = data.map(
            lambda a, b, c, d, e, f, g: (a, b, c, d, e, tf.RaggedTensor.from_tensor(f), tf.RaggedTensor.from_tensor(g)))
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        data = data.map(lambda a, b, c, d, e, f, g: (a, b, c, d, e, tf.squeeze(f, 1), tf.squeeze(g, 1)))

        return data

    def step_eval(self, user):
        n_items = self._nitems
        ui_dict = self._ui_dict
        iu_dict = self._iu_dict

        def sample(u, i):
            u_pos = ui_dict[u]
            i_pos = iu_dict[i]

            u_ratings = np.zeros((1, self._nitems))
            i_ratings = np.zeros((1, self._nusers))
            u_ratings[0, u_pos] = 1.0
            i_ratings[0, i_pos] = 1.0

            u_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['USER_ID'] == self._private_users[u]][
                    'TOKENS_POSITION'].tolist()
            i_review_tokens = \
                self._train_reviews_tokens[self._train_reviews_tokens['ITEM_ID'] == self._private_items[i]][
                    'TOKENS_POSITION'].tolist()

            return u, i, 1.0, u_ratings, i_ratings, u_review_tokens, i_review_tokens

        for item in range(n_items):
            yield sample(user, item)

    def pipeline_eval(self, user, batch_size):
        data = tf.data.Dataset.from_generator(
            generator=self.step_eval,
            output_types=(tf.int64, tf.int64, tf.float32, tf.float32, tf.float32, tf.int64, tf.int64),
            args=(user,)
        )
        data = data.map(lambda a, b, c, d, e, f, g: (a, b, c, d, e, tf.expand_dims(f, 0), tf.expand_dims(g, 0)))
        data = data.map(
            lambda a, b, c, d, e, f, g: (a, b, c, d, e, tf.RaggedTensor.from_tensor(f), tf.RaggedTensor.from_tensor(g)))
        data = data.batch(batch_size=batch_size)
        data = data.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        data = data.map(lambda a, b, c, d, e, f, g: (a, b, c, d, e, tf.squeeze(f, 1), tf.squeeze(g, 1)))

        return data
