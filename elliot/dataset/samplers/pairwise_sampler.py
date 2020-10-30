import typing as t


class Sampler:
    def __init__(self, ratings,
                 random,
                 sample_negative_items_empirically: bool = True
                 ):
        self._ratings = ratings
        self._random = random
        self._sample_negative_items_empirically: bool = sample_negative_items_empirically
        self._users = list(self._ratings.keys())
        self._items = list({k for a in self._ratings.values() for k in a.keys()})

    def step(self, events: int):
        r_int = self._random.randint
        n_users = len(self._users)
        n_items = len(self._items)
        users = self._users
        items = self._items
        ratings = self._ratings

        for _ in range(events):
            u = users[r_int(n_users)]
            ui = set(ratings[u].keys())
            lui = len(ui)
            if lui == n_items:
                continue
            i = list(ui)[r_int(lui)]

            j = items[r_int(n_items)]
            while j in ui:
                j = items[r_int(n_items)]

            yield u, i, j
