"""
This is the implementation of the Item Coverage metric.
It directly proceeds from a system-wise computation, and it considers all the users at the same time.
"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from elliot.evaluation.metrics.base_metric import BaseMetric


class ItemCoverage(BaseMetric):
    r"""
    Item Coverage

    This class represents the implementation of the Item Coverage recommendation metric.

    For further details, please refer to the `book <https://link.springer.com/10.1007/978-1-4939-7131-2_110158>`_

    Note:
         The simplest measure of catalog coverage is the percentage of all items that can ever be recommended.
         This measure can be computed in many cases directly given the algorithm and the input data set.

    To compute the metric, add it to the config file adopting the following pattern:

    .. code:: yaml

        simple_metrics: [ItemCoverage]
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
        self._cutoff = self._evaluation_objects.cutoff

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "ItemCoverage"

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Item Coverage
        """
        return len({i[0] for u_r in self._recommendations.values() for i in u_r[:self._cutoff]})
