import typing as t
import numpy as np

class Precision:
    def __init__(self, recommendations, cutoff, relevant_items):
        self.recommendations, self.cutoff, self.relevant_items = recommendations, cutoff, relevant_items

    @staticmethod
    def name():
        return "Precision"

    @staticmethod
    def __user_precision(user_recommendations, cutoff, user_relevant_items):
        return sum([1 for i in user_recommendations if i[0] in user_relevant_items]) / cutoff

    def eval(self):
        return np.average(
            Precision.__user_precision(u_r, self.cutoff, self.relevant_items[u])
            for u, u_r in self.recommendations.items()
        )
