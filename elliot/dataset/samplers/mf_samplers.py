"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers.base_sampler import TraditionalSampler


class BPRMFSampler(TraditionalSampler):
    def __init__(self, indexed_ratings, seed=42):
        super().__init__(seed, indexed_ratings)

        self._freq_users = np.zeros(self._nusers, dtype=np.int64)
        self._freq_items = np.zeros(self._nitems, dtype=np.int64)

    @property
    def freq_users(self):
        return dict(enumerate(self._freq_users))

    @property
    def freq_items(self):
        return dict(enumerate(self._freq_items))

    def _sample(self, **kwargs):
        users = self._r_int(0, self._nusers, size=self.events)
        self._freq_users[users] += 1

        pos_items = np.empty(self.events, dtype=np.int64)
        neg_items = np.empty(self.events, dtype=np.int64)

        iter_data = tqdm(enumerate(users), total=self.events, desc="Sampling", leave=False)

        for idx, u in iter_data:
            if self._lui_dict[u] == self._nitems:
                self._freq_users[u] -= 1
                while u in users:
                    u = self._r_int(self._nusers)
                self._freq_users[u] += 1

            ui = self._ui_dict[u]
            lui = self._lui_dict[u]

            i = ui[self._r_int(lui)]
            pos_items[idx] = i
            self._freq_items[i] += 1

            while True:
                j = self._r_int(self._nitems)
                if j not in ui:
                    neg_items[idx] = j
                    self._freq_items[j] += 1
                    break

        return users[:, None], pos_items[:, None], neg_items[:, None]
        # u = self._r_int(self._nusers)
        # self._freq_users[u] += 1
        # ui = self._ui_dict[u]
        # lui = self._lui_dict[u]
        # if lui == self._nitems:
        #     self._freq_users[u] -= 1
        #     return self._sample()
        # i = ui[self._r_int(lui)]
        # self._freq_items[i] += 1
        #
        # j = self._r_int(self._nitems)
        # while j in ui:
        #     j = self._r_int(self._nitems)
        # self._freq_items[j] += 1
        # return u, i, j
        # def prepare_output(self, *args):
        #     return (r[:, None] for r in args)

    """def step(self, events: int, batch_size: int):
        r_int = np.random.randint
        n_users = self._nusers
        n_items = self._nitems
        ui_dict = self._ui_dict
        lui_dict = self._lui_dict

        def sample():
            u = r_int(n_users)
            self._freq_users[u] += 1
            ui = ui_dict[u]
            lui = lui_dict[u]
            if lui == n_items:
                self._freq_users[u] -= 1
                sample()
            i = ui[r_int(lui)]
            self._freq_items[i] += 1

            j = r_int(n_items)
            while j in ui:
                j = r_int(n_items)
            self._freq_items[j] += 1
            return u, i, j

        for batch_start in range(0, events, batch_size):
            bui, bii, bij = map(np.array,
                                zip(*[sample() for _ in
                                      range(batch_start, min(batch_start + batch_size, events))]))
            yield bui[:, None], bii[:, None], bij[:, None]"""


class MFSampler(TraditionalSampler):
    def __init__(self, indexed_ratings, sparse_matrix, seed=42):
        super().__init__(seed, indexed_ratings)

        ratings = sparse_matrix.nonzero()
        self.rating_users = ratings[0]
        self.rating_items = ratings[1]

        self._users, self.idx_start, self.count = np.unique(self.rating_users, return_counts=True, return_index=True)
        self.m = 0
        self.num_neg = lambda n: self.m

    def _sample(self, **kwargs):
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
        users = []
        items = []
        labels = []
        all_items = np.array(self._items, dtype=np.int64)

        iter_data = tqdm(self._users, total=len(self._users), desc="Sampling", leave=False)

        for u in iter_data:
            start, end = self.idx_start[u], self.idx_start[u] + self.count[u]
            pos_u = self.rating_items[start:end]

            # neg_candidates = np.setdiff1d(all_items, pos_u, assume_unique=True)
            # neg_u = self._r_choice(neg_candidates, self.m * len(pos_u), replace=True)
            neg_samples = []
            while len(neg_samples) < self.num_neg(len(pos_u)):
                candidate_neg = self._r_choice(all_items, self.num_neg(len(pos_u)) - len(neg_samples), replace=True)
                mask = ~np.isin(candidate_neg, pos_u)
                neg_samples.extend(candidate_neg[mask])
            neg_u = np.array(neg_samples[:self.num_neg(len(pos_u))], dtype=np.int64)

            users.extend([u] * (len(pos_u) + len(neg_u)))
            items.extend(np.concatenate([pos_u, neg_u]))
            labels.extend(np.concatenate([
                np.ones(len(pos_u), dtype=np.int8),
                np.zeros(len(neg_u), dtype=np.int8)
            ]))

        training_matrix = np.column_stack([users, items, labels])
        indices = self._r_perm(len(training_matrix))
        u, i, pos = training_matrix[indices].T
        return u, i, pos
        # time_start = time.time()
        #
        # def user_training_matrix(u):
        #     pos_u = self.rating_items[self.idx_start[u]:self.idx_start[u] + self.count[u]]
        #     neg_u = np.setdiff1d(np.array(self._items), pos_u, assume_unique=True)
        #     sampled_neg_u = self._r_choice(neg_u, self.m * len(pos_u), replace=True)
        #     return np.c_[np.repeat(u, len(pos_u) + len(sampled_neg_u)), np.r_[
        #         np.c_[pos_u, np.ones(len(pos_u), dtype=int)], np.c_[
        #             sampled_neg_u, np.zeros(len(sampled_neg_u), dtype=int)]]]
        #
        # training_matrix = np.concatenate([user_training_matrix(u) for u in self._users])
        #
        # samples_indices = random.sample(range(training_matrix.shape[0]), training_matrix.shape[0])
        # self._samples = training_matrix[samples_indices]
        # print(f"Sampling has taken {round(time.time() - time_start, 2)} seconds")
        #for start in range(0, training_matrix.shape[0], batch_size):
        #    yield training_matrix[start:min(start + batch_size, training_matrix.shape[0])]

    # def _sample(self, bs, bsize):
    #     u, i, pos = self._samples[bs:bs + bsize].T
    #     return u, i, pos


class MFSamplerRendle(MFSampler):
    def __init__(self, indexed_ratings, sparse_matrix, seed=42):
        super().__init__(indexed_ratings, sparse_matrix, seed)
        self.num_neg = lambda n: self.m * n

# class Sampler(TraditionalSampler):
#     def __init__(self, indexed_ratings, m, sparse_matrix, seed=42):
#         super().__init__(indexed_ratings, seed)
#         #np.random.seed(seed)
#         #random.seed(seed)
#         self._sparse = sparse_matrix
#         """self._indexed_ratings = indexed_ratings
#         self._users = list(self._indexed_ratings.keys())
#         self._nusers = len(self._users)
#         self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
#         self._nitems = len(self._items)
#         self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
#         self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""
#         self._m = m
#
#     def initialize(self):
#         nonzero = self._sparse.nonzero()
#         pos = list(zip(*nonzero, np.ones(len(nonzero[0]), dtype=np.int32)))
#
#         neg = list()
#         for u, i, _ in pos:
#             neg_samples = random.sample(range(self._nitems), self._m)
#             neg += list(zip(np.ones(len(neg_samples), dtype=np.int32) * u, neg_samples,
#                             np.zeros(len(neg_samples), dtype=np.int32)))
#             pass
#             # for _ in range(self._m):
#             #     neg.add((u, r_int(n_items), 0))
#
#         # samples = list(pos)
#         samples = pos + neg
#         self._samples = random.sample(samples, len(samples))
#
#         """nonzero = self._sparse.nonzero()
#         pos_u = np.array(nonzero[0], dtype=np.int32)
#         pos_i = np.array(nonzero[1], dtype=np.int32)
#         pos_y = np.ones(len(pos_u), dtype=np.int32)
#         pos = np.stack([pos_u, pos_i, pos_y], axis=1)
#
#         n_pos = len(pos_u)
#         neg_u = np.repeat(pos_u, self._m)
#         neg_i = np.random.randint(0, self._nitems, size=n_pos * self._m, dtype=np.int32)
#         neg_y = np.zeros(n_pos * self._m, dtype=np.int32)
#         neg = np.stack([neg_u, neg_i, neg_y], axis=1)
#
#         samples = np.concatenate([pos, neg], axis=0)
#         np.random.shuffle(samples)
#
#         self._samples = samples"""
#
#     def _sample(self, bs, bsize):
#         return self._samples[bs:bs + bsize]
#         return tuple(list(c) for c in zip(*samples))
#
#     """def step(self, batch_size):
#         r_int = np.random.randint
#         n_users = self._nusers
#         n_items = self._nitems
#         ui_dict = self._ui_dict
#         lui_dict = self._lui_dict
#
#         # def sample_pos(u):
#         #     ui = ui_dict[u]
#         #     lui = lui_dict[u]
#         #     if lui == n_items:
#         #         return None
#         #     return ui[r_int(lui)]
#         # pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}
#         nonzero = self._sparse.nonzero()
#         pos = list(zip(*nonzero,np.ones(len(nonzero[0]), dtype=np.int32)))
#
#         neg = list()
#         for u, i, _ in pos:
#             neg_samples = random.sample(range(n_items), self._m)
#             neg += list(zip(np.ones(len(neg_samples), dtype=np.int32) * u, neg_samples, np.zeros(len(neg_samples), dtype=np.int32)))
#             pass
#             # for _ in range(self._m):
#             #     neg.add((u, r_int(n_items), 0))
#
#         # samples = list(pos)
#         samples = pos + neg
#         samples = random.sample(samples, len(samples))
#
#         # def sample():
#         #     u = r_int(n_users)
#         #     ui = ui_dict[u]
#         #     lui = lui_dict[u]
#         #     if lui == n_items:
#         #         sample()
#         #     i = ui[r_int(lui)]
#         #
#         #     j = r_int(n_items)
#         #     while j in ui:
#         #         j = r_int(n_items)
#         #     return u, i, j
#
#         # for sample in zip(samples):
#         #     yield
#
#         for start in range(0, len(samples), batch_size):
#             # u, i, b = samples[start:min(start + batch_size, len(samples))]
#             yield samples[start:min(start + batch_size, len(samples))]"""

# class Sampler(TraditionalSampler):
#     def __init__(self, indexed_ratings, m, sparse_matrix, seed=42):
#         super().__init__(indexed_ratings, seed)
#         #np.random.seed(seed)
#         #random.seed(seed)
#         self._sparse = sparse_matrix
#         """self._indexed_ratings = indexed_ratings
#         self._users = list(self._indexed_ratings.keys())
#         self._nusers = len(self._users)
#         self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
#         self._nitems = len(self._items)
#         self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
#         self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}"""
#         self._m = m
#
#     def initialize(self):
#         nonzero = self._sparse.nonzero()
#         pos = list(zip(*nonzero, np.ones(len(nonzero[0]), dtype=np.int32)))
#
#         neg = set()
#         for u, i, _ in pos:
#             for _ in range(self._m):
#                 neg.add((u, self._r_int(self._nitems), 0))
#
#         # samples = list(pos)
#         samples = pos + list(neg)
#         self._samples = random.sample(samples, len(samples))
#
#     def _sample(self, bs, bsize):
#         return self._samples[bs:bs + bsize]
#         return tuple(list(c) for c in zip(*samples))
#
#     """def step(self, batch_size):
#         # t1 = time()
#         r_int = np.random.randint
#         n_users = self._nusers
#         n_items = self._nitems
#         ui_dict = self._ui_dict
#         lui_dict = self._lui_dict
#
#         # pos = {(u, i, 1) for u, items in ui_dict.items() for i in items}
#
#         nonzero = self._sparse.nonzero()
#         pos = list(zip(*nonzero, np.ones(len(nonzero[0]), dtype=np.int32)))
#
#         neg = set()
#         for u, i, _ in pos:
#             for _ in range(self._m):
#                 neg.add((u, r_int(n_items), 0))
#
#         # samples = list(pos)
#         samples = pos + list(neg)
#         samples = random.sample(samples, len(samples))
#         # t2 = time()
#         # print('Epoch sampling [%.1f s]', t2 - t1)
#
#         for start in range(0, len(samples), batch_size):
#             yield samples[start:min(start + batch_size, len(samples))]"""