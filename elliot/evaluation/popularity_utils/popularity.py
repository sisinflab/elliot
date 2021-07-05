"""
Module description:
This module provides a popularity class based on number of users who have experienced an item (user-item repetitions in
the dataset are counted once)
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

import operator

import typing as t


class Popularity(object):
    def __init__(self, data, pop_ratio=0.8):
        self._data = data
        self._pop_items = {}
        self._sorted_pop_items = {}
        self._short_head = []
        self._long_tail = []
        self._pop_ratio = pop_ratio

    def get_pop_items(self):
        if not self._pop_items:
            self._pop_items = {self._data.private_items[p]: pop for p, pop in
                               enumerate(self._data.sp_i_train.astype(bool).sum(axis=0).tolist()[0])}
        return self._pop_items

    def get_sorted_pop_items(self):
        if (not self._pop_items) or (not self._sorted_pop_items):
            self.get_pop_items()
            self._sorted_pop_items = dict(sorted(self._pop_items.items(), key=operator.itemgetter(1), reverse=True))
        return self._sorted_pop_items

    def get_short_head(self):
        if not self._short_head:
            self.get_sorted_pop_items()
            short_head_limit = self._data.transactions * self._pop_ratio
            self._short_head = []
            for i, pop in self._sorted_pop_items.items():
                self._short_head.append(i)
                short_head_limit -= pop
                if short_head_limit <= 0:
                    break
        return self._short_head

    def get_long_tail(self):
        if not self._long_tail:
            self.get_short_head()
            self._long_tail = [i for i in self._sorted_pop_items.keys() if i not in self._short_head]
        return self._long_tail

    def get_custom_pop_obj(self, pop_ratio=.8):
        return Popularity(self._data, pop_ratio)


