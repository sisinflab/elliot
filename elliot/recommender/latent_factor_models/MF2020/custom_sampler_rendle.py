"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
import random
import time


class Sampler:
    def __init__(self, indexed_ratings, m, sparse_matrix, seed):
        np.random.seed(seed)
        random.seed(seed)
        self._sparse = sparse_matrix
        self._indexed_ratings = indexed_ratings
        self._users = list(self._indexed_ratings.keys())
        self._nusers = len(self._users)
        self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
        self._nitems = len(self._items)
        self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self._m = m
        self._nonzero = self._sparse.nonzero()
        self._num_pos_examples = len(self._nonzero[0])
        self._positive_pairs = list(zip(*self._nonzero, np.ones(len(self._nonzero[0]), dtype=np.int32)))

    def step(self, batch_size):
        """Converts a list of positive pairs into a two class dataset.
        Args:
          positive_pairs: an array of shape [n, 2], each row representing a positive
            user-item pair.
          num_negatives: the number of negative items to sample for each positive.
        Returns:
          An array of shape [n*(1 + num_negatives), 3], where each row is a tuple
          (user, item, label). The examples are obtained as follows:
          To each (user, item) pair in positive_pairs correspond:
          * one positive example (user, item, 1)
          * num_negatives negative examples (user, item', 0) where item' is sampled
            uniformly at random.
        """
        time_start = time.time()
        r_int = np.random.randint
        num_items = self._nitems
        num_negatives = self._m
        num_pos_examples = self._num_pos_examples
        positive_pairs = self._positive_pairs

        training_matrix = np.empty([num_pos_examples * (1 + num_negatives), 3],
                                   dtype=np.int32)
        index = 0

        for pos_index in range(num_pos_examples):
            u = positive_pairs[pos_index][0]
            i = positive_pairs[pos_index][1]

            # Treat the rating as a positive training instance
            training_matrix[index] = [u, i, 1]
            index += 1

            # Add N negatives by sampling random items.
            # This code does not enforce that the sampled negatives are not present in
            # the training data. It is possible that the sampling procedure adds a
            # negative that is already in the set of positives. It is also possible
            # that an item is sampled twice. Both cases should be fine.
            for _ in range(num_negatives):
                j = r_int(num_items)
                training_matrix[index] = [u, j, 0]
                index += 1
        # neg = set()
        # for u, i, _ in pos:
        #     for _ in range(self._m):
        #         neg.add((u, r_int(n_items), 0))
        #
        # samples = list(pos)
        # samples += list(neg)
        samples_indices = random.sample(range(training_matrix.shape[0]), training_matrix.shape[0])
        training_matrix = training_matrix[samples_indices]
        print(f"Sampling has taken {round(time.time()-time_start, 2)} seconds")
        for start in range(0, training_matrix.shape[0], batch_size):
            yield training_matrix[start:min(start + batch_size, training_matrix.shape[0])]
