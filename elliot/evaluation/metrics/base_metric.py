"""
This is the implementation of the Precision metric.
It proceeds from a user-wise computation and averages the values over the users.

"""

from typing import List, Dict, Tuple
from types import SimpleNamespace
from abc import ABC
import numpy as np

from elliot.namespace import ExperimentConfig, RecommenderConfig


class BaseMetric(ABC):
    """
    This class represents the implementation of the Precision recommendation metric.
    Passing 'Precision' to the metrics list will enable the computation of the metric.
    """

    needs_full_recommendations: bool = False

    def __init__(
        self,
        recommendations: Dict[int, List[Tuple[int, float]]],
        config: ExperimentConfig,
        params: RecommenderConfig,
        eval_objects: SimpleNamespace,
        additional_data=None
    ):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param cutoff: numerical threshold to limit the recommendation list
        :param relevant_items: list of relevant items (binary) per user in the form {user: [item1,...]}
        """
        self._recommendations = recommendations
        self._config = config
        self._params = params
        self._evaluation_objects = eval_objects
        self._additional_data = additional_data

    @property
    def name(self):
        return self.__class__.__name__

    def eval(self):
        return np.average(list(self.eval_user_metric().values()) or 0)

    def eval_user_metric(self):
        return {}

    def get(self):
        return [self]
