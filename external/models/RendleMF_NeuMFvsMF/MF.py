"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import pickle
import time
from tqdm import tqdm

import numpy as np

from elliot.dataset.samplers import pointwise_pos_neg_sampler as ps
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from .MF_model import MFModel

np.random.seed(42)


class MF(RecMixin, BaseRecommenderModel):
    r"""
    Matrix Factorization (implementation from "Neural Collaborative Filtering vs. Matrix Factorization Revisited")

    For further details, please refer to the `paper <https://dl.acm.org/doi/pdf/10.1145/3383313.3412488>`_

    Args:
        factors: Number of latent factors
        lr: Learning rate
        bias_regularization: Regularization coefficient for the bias
        user_regularization: Regularization coefficient for user latent factors
        positive_item_regularization: Regularization coefficient for positive item latent factors
        negative_item_regularization: Regularization coefficient for negative item latent factors
        update_negative_item_factors:
        update_users:
        update_items:
        update_bias:


    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        MF:
          meta:
            save_recs: True
          epochs: 10
          factors: 10
          lr: 0.001
          reg: 0.0025
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "f", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.05, None, None),
            ("_regularization", "reg", "reg", 0, None, None)
        ]
        self.autoset_params()

        self._ratings = self._data.train_dict
        self._sampler = ps.Sampler(self._data.i_train_dict)

        self._batch_size = 1

        self._model = MFModel(self._factors,
                              self._data,
                              self._learning_rate,
                              self._regularization)

    def get_recommendations(self, k: int = 10):
        predictions_top_k_val = {}
        predictions_top_k_test = {}

        recs_val, recs_test = self.process_protocol(k)

        predictions_top_k_val.update(recs_val)
        predictions_top_k_test.update(recs_test)

        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, *args):
        return {u: self._model.get_user_predictions(u, mask, k) for u in self._data.train_dict.keys()}

    def predict(self, u: int, i: int):
        """
        Get prediction on the user item pair.

        Returns:
            A single float vaue.
        """
        return self._model.predict(u, i)

    @property
    def name(self):
        return "MF" \
               + "_e:" + str(self._epochs) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        print(f"Transactions: {self._data.transactions}")

        for it in range(self._epochs):
            print(f"\n********** Iteration: {it + 1}")
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss/steps:.5f}'})
                    t.update()

            self.evaluate(it, loss)

    def restore_weights(self):
        try:
            with open(self._saving_filepath, "rb") as f:
                self._model.set_model_state(pickle.load(f))
            print(f"Model correctly Restored")

            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)
            self._results.append(result_dict)

            print("******************************************")
            if self._save_recs:
                store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
            return True

        except Exception as ex:
            print(f"Error in model restoring operation! {ex}")

        return False
