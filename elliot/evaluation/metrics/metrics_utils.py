
import typing as t

from .base_metric import BaseMetric


class ProxyMetric(BaseMetric):
    """

    """
    def __init__(self, name="ProxyMetric", val=0, needs_full_recommendations=False):
        self._name =name
        self._val = val
        self._needs_full_recommendations = needs_full_recommendations

    def name(self):
        return self._name

    def eval(self):
        return self._val

    def needs_full_recommendations(self):
        return self._needs_full_recommendations


class ProxyStatisticalMetric(BaseMetric):
    """

    """
    def __init__(self, name="ProxyMetric", val=0, user_val=0, needs_full_recommendations=False):
        self._name =name
        self._val = val
        self._user_val = user_val
        self._needs_full_recommendations = needs_full_recommendations

    def name(self):
        return self._name

    def eval(self):
        return self._val

    def eval_user_metric(self):
        return self._user_val

    def needs_full_recommendations(self):
        return self._needs_full_recommendations
