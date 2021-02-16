"""
This is the implementation of the User Coverage metric.
It directly proceeds from a system-wise computation, and it considers all the users at the same time.
"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.evaluation.metrics.base_metric import BaseMetric


class UserCoverage(BaseMetric):
    r"""
    This class represents the implementation of the User Coverage recommendation metric.
    Passing 'UserCoverage' to the metrics list will enable the computation of the metric.

    .. _ItemCoverage: "Recommender systems handbook. Springer, Berlin"
    Ricci F, Rokach L, Shapira B, Kantor P. 2015

    Note:
          The proportion of users or user interactions for which the system can recommend items. In many applications
          the recommender may not provide recommendations for some users due to, e.g. low confidence in the accuracy
          of predictions for that user.


    """

    def __init__(self, recommendations, config, params, eval_objects):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param config: SimpleNameSpace that represents the configuration of the experiment
        :param params: Parameters of the model
        :param eval_objects: list of objects that may be useful for the computation of the different metrics
        """
        super().__init__(recommendations, config, params, eval_objects)

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "UserCoverage"

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of User Coverage
        """
        return sum([1 if len(u_r) > 0 else 0 for u_r in self._recommendations.values()])
