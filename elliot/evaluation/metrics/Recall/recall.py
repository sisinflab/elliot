import numpy as np

class Recall:
    def __init__(self, recommendations, cutoff, relevant_items):
        self.recommendations, self.cutoff, self.relevant_items = recommendations, cutoff, relevant_items

    @staticmethod
    def name():
        return "Recall"

    @staticmethod
    def __user_recall(user_recommendations, cutoff, user_relevant_items):
        #TODO check formula
        return sum([1 for i in user_recommendations if i[0] in user_relevant_items]) / \
               min(len(user_relevant_items), cutoff)

    def eval(self):
        return np.average(
            Recall.__user_recall(u_r, self.cutoff, self.relevant_items[u])
            for u, u_r in self.recommendations.items()
        )
