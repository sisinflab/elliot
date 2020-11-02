"""
This is the implementation of the Item Coverage metric.
It directly proceeds from a system-wise computation, and it considers all the users at the same time.
"""

__version__ = '0.1'
__author__ = 'XXX'


class ItemCoverage:
    """
    This class represents the implementation of the Item Coverage recommendation metric.
    Passing 'ItemCoverage' to the metrics list will enable the computation of the metric.
    """

    def __init__(self, recommendations, params):
        """
        Constructor
        :param recommendations: list of recommendations in the form {user: [(item1,value1),...]}
        :param cutoff: numerical threshold to limit the recommendation list
        :param relevant_items: list of relevant items (binary) per user in the form {user: [item1,...]}
        """
        self.recommendations = recommendations
        self.cutoff = params.k

    @staticmethod
    def name():
        """
        Metric Name Getter
        :return: returns the public name of the metric
        """
        return "Item-Coverage"

    def eval(self):
        """
        Evaluation function
        :return: the overall averaged value of Item Coverage
        """
        return len({i[0] for u_r in self.recommendations.values() for i in u_r[:self.cutoff]})
