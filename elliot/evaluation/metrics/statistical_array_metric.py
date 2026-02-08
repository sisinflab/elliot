"""
This is the implementation of the Precision metric.
It proceeds from a user-wise computation, and average the values over the users.
"""


from abc import ABCMeta, abstractmethod


class StatisticalMetric(metaclass=ABCMeta):
    """
    This class represents the implementation of the Precision recommendation metric.
    Passing 'Precision' to the metrics list will enable the computation of the metric.
    """

    @abstractmethod
    def eval_user_metric(self):
        pass

    @classmethod
    def __subclasshook__(cls, C):
        if cls is StatisticalMetric:
            if any("eval_user_metric" in B.__dict__ for B in C.__mro__):
                return True
        return NotImplemented
