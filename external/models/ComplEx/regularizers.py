"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Pasquale Minervini'
__email__ = 'p.minervini@ucl.ac.uk'

from abc import ABC, abstractmethod

import tensorflow as tf

from typing import List


class Regularizer(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self,
                 factors: List[tf.Tensor]):
        raise NotImplementedError


class F2(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[tf.Tensor]):
        norm = sum(tf.reduce_sum(f ** 2) for f in factors)
        return norm / factors[0].shape[0]


class N3(Regularizer):
    def __init__(self):
        super().__init__()

    def __call__(self,
                 factors: List[tf.Tensor]):
        norm = sum(tf.reduce_sum(tf.abs(f) ** 3) for f in factors)
        return norm / factors[0].shape[0]
