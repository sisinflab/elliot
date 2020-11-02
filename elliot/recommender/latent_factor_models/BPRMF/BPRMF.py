import time
import typing as t
import numpy as np
from recommender.base_recommender_model import BaseRecommenderModel
from dataset.dataset import DataSet
from dataset.samplers import pairwise_sampler as ps
from evaluation.evaluator import Evaluator
from utils.write import store_recommendation


class MF(object):
    """
    Simple Matrix Factorization class
    """

    def __init__(self, F, ratings, random, *args):
        self._factors = F
        self._ratings = ratings
        self._random = random
        self.initialize(*args)

    def initialize(self, loc: float = 0, scale: float = 0.1):
        """
        This function initialize the data model
        :param loc:
        :param scale:
        :return:
        """
        self._users = list(self._ratings.keys())
        self._items = list({k for a in self._ratings.values() for k in a.keys()})
        self._private_users = {p: u for p, u in enumerate(self._users)}
        self._public_users = {v: k for k, v in self._private_users.items()}
        self._private_items = {p: i for p, i in enumerate(self._items)}
        self._public_items = {v: k for k, v in self._private_items.items()}

        self._global_bias = 0

        "same parameters as np.randn"
        self._user_bias = np.zeros(len(self._users))
        self._item_bias = np.zeros(len(self._items))
        self._user_factors = \
            self._random.normal(loc=loc, scale=scale, size=(len(self._users), self._factors))
        self._item_factors = \
            self._random.normal(loc=loc, scale=scale, size=(len(self._items), self._factors))
        self._transactions = sum(len(v) for v in self._ratings.values())

    @property
    def name(self):
        return "MF"

    def get_transactions(self):
        return self._transactions

    def predict(self, user, item):
        return self._global_bias + self._item_bias[self._public_items[item]] \
               + self._user_factors[self._public_users[user]] @ self._item_factors[self._public_items[item]]

    def get_user_recs(self, user, k):
        arr = self._item_bias + self._item_factors @ self._user_factors[self._public_users[user]]
        top_k = arr.argsort()[-(len(self._ratings[user].keys()) + k):][::-1]
        top_k_2 = [(self._private_items[i], arr[i]) for p, i in enumerate(top_k)
                   if (self._private_items[i] not in self._ratings[user].keys())]
        top_k_2 = top_k_2[:k]
        return top_k_2

    def get_user_recs_argpartition(self, user: int, k: int):
        user_items = self._ratings[user].keys()
        safety_k = len(user_items)+k
        predictions = self._item_bias +  self._item_factors  @ self._user_factors[self._public_users[user]]
        partially_ordered_preds_indices = np.argpartition(predictions, -safety_k)[-safety_k:]
        partially_ordered_preds_values = predictions[partially_ordered_preds_indices]
        partially_ordered_preds_ids = [self._private_items[x] for x in partially_ordered_preds_indices]

        top_k = partially_ordered_preds_values.argsort()[::-1]
        top_k_2 = [(partially_ordered_preds_ids[i], partially_ordered_preds_values[i]) for p, i in enumerate(top_k)
                   if (partially_ordered_preds_ids[i] not in user_items)]
        top_k_2 = top_k_2[:k]
        return top_k_2


    def get_user_bias(self, user: int):

        return self._user_bias[self._public_users[user]]

    def get_item_bias(self, item: int):

        return self._item_bias[self._public_items[item]]

    def get_user_factors(self, user: int):

        return self._user_factors[self._public_users[user]]

    def get_item_factors(self, item: int):

        return self._item_factors[self._public_items[item]]

    def set_user_bias(self, user: int, v: float):

        self._user_bias[self._public_users[user]] = v

    def set_item_bias(self, item: int, v: float):

        self._item_bias[self._public_items[item]] = v

    def set_user_factors(self, user: int, v: float):

        self._user_factors[self._public_users[user]] = v

    def set_item_factors(self, item: int, v: float):

        self._item_factors[self._public_items[item]] = v


class BPRMF(BaseRecommenderModel):

    def __init__(self, config, params, *args, **kwargs):
        super().__init__(config, params, *args, **kwargs)
        np.random.seed(42)

        self._data = DataSet(config, params)
        self._config = config
        self._params = params
        self._num_items = self._data.num_items
        self._num_users = self._data.num_users
        self._random = np.random
        self._sample_negative_items_empirically = True
        print(self.params)
        self._factors = self.params.embed_k
        self._learning_rate = self.params.lr
        self._bias_regularization = self.params.l_b
        self._user_regularization = self.params.lr / 20
        self._positive_item_regularization = self.params.lr / 20
        self._negative_item_regularization = self.params.lr / 200
        self._update_negative_item_factors = True
        self._update_users = True
        self._update_items = True
        self._update_bias = True

        self._ratings = self._data.train_dataframe_dict
        self._datamodel = MF(self._factors, self._ratings, self._random)
        self._sampler = ps.Sampler(self._ratings, self._random, self._sample_negative_items_empirically)

        self._iteration = 0

        self.evaluator = Evaluator(self._data)

    def get_recommendations(self, k: int = 100):
        return {u: self._datamodel.get_user_recs(u, k) for u in self._ratings.keys()}

    def predict(self, u: int, i: int):
        """
        Get prediction on the user item pair.

        Returns:
            A single float vaue.
        """
        return self._datamodel.predict(u, i)

    def _get_parameters(self):
        return f"_F{self._factors}_I{self._iteration+1}_L{str(self._learning_rate).replace('.','-')}"

    @property
    def name(self):
        return "BPR" + self._datamodel.name

    def train_step(self):
        start_it = time.perf_counter()
        print()
        print("Sampling...")
        samples = self._sampler.step(self.num_pos_events)
        start = time.perf_counter()
        print(f"Sampled in {round(start-start_it, 2)} seconds")
        # print(f"Training samples: {len(samples)}")
        start = time.perf_counter()
        print("Computing..")
        for u, i, j in samples:
            self.update_factors(u, i, j)
        t2 = time.perf_counter()
        print(f"Computed and updated in {round(t2-start, 2)} seconds")
        # if (it + 1) % 1 == 0:
        #     name = f"BPRMF_F{self._factors}_I{it+1}_L{str(self._learning_rate).replace('.','-')}"
        #     print("Printing..")
        #     self.print_recs("../recs/" + name + ".tsv", 10)

    def train(self, num_iters: int = 10):
        self.num_pos_events = self._datamodel.get_transactions()
        print(f"Transactions: {self.num_pos_events}")
        for it in range(num_iters):
            print()
            print(f"********** Iteration: {it + 1}")
            self._iteration = it

            self.train_step()
            recs = self.get_recommendations(self._params.k)
            self.evaluator.eval(recs)

            if not (it+1) % 10:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}_{it+1}.tsv")

    def update_factors(self, u: int, i: int, j: int):
        user_factors = self._datamodel.get_user_factors(u)
        item_factors_i = self._datamodel.get_item_factors(i)
        item_factors_j = self._datamodel.get_item_factors(j)
        item_bias_i = self._datamodel.get_item_bias(i)
        item_bias_j = self._datamodel.get_item_bias(j)

        z = 1/(1 + np.exp(self.predict(u, i)-self.predict(u, j)))
        # update bias i
        d_bi = (z - self._bias_regularization*item_bias_i)
        self._datamodel.set_item_bias(i, item_bias_i
                                 + (self._learning_rate * d_bi))

        # update bias j
        d_bj = (-z - self._bias_regularization*item_bias_j)
        self._datamodel.set_item_bias(j, item_bias_j
                                 + (self._learning_rate * d_bj))

        # update user factors
        d_u = ((item_factors_i - item_factors_j)*z - self._user_regularization*user_factors)
        self._datamodel.set_user_factors(u, user_factors +
                                    (self._learning_rate * d_u))

        # update item i factors
        d_i = (user_factors*z - self._positive_item_regularization*item_factors_i)
        self._datamodel.set_item_factors(i, item_factors_i
                                    + (self._learning_rate * d_i))

        # update item j factors
        d_j = (-user_factors*z - self._negative_item_regularization*item_factors_j)
        self._datamodel.set_item_factors(j, item_factors_j
                                    + (self._learning_rate * d_j))
