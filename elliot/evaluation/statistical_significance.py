"""
Module description:
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alejandro Bellogín'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alejandro.bellogin@uam.es'

from scipy import stats
import typing as t
import numpy as np


class PairedTTest:
    @staticmethod
    def common_users(arr_0: t.Dict[int, float], arr_1:  t.Dict[int, float]):
        return list(arr_0.keys() & arr_1.keys())

    @staticmethod
    def compare(arr_0: t.Dict[int, float], arr_1:  t.Dict[int, float], users: t.List[int]):
        list_0 = list(map(arr_0.get, users))
        list_1 = list(map(arr_1.get, users))
        return stats.ttest_rel(list_0, list_1)[1]


class WilcoxonTest:
    @staticmethod
    def common_users(arr_0: t.Dict[int, float], arr_1:  t.Dict[int, float]):
        return list(arr_0.keys() & arr_1.keys())

    @staticmethod
    def compare(arr_0: t.Dict[int, float], arr_1:  t.Dict[int, float], users: t.List[int]):
        list_0 = list(map(arr_0.get, users))
        list_1 = list(map(arr_1.get, users))
        return stats.wilcoxon(list_0, list_1)[1] if any(np.array(list_0) - np.array(list_1)) else np.nan


