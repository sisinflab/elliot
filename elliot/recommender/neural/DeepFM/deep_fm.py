"""
Module description:

"""


__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta, Antonio Ferrara'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it,' \
            'daniele.malitesta@poliba.it, antonio.ferrara@poliba.it'

import pickle
from ast import literal_eval as make_tuple

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.neural.DeepFM.deep_fm_model import DeepFMModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class DeepFM(RecMixin, BaseRecommenderModel):
    r"""
        DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

        For further details, please refer to the `paper <https://arxiv.org/abs/1703.04247>`_

        Args:
            factors: Number of factors dimension
            lr: Learning rate
            l_w: Regularization coefficient
            hidden_neurons: List of units for each layer
            hidden_activations: List of activation functions

        To include the recommendation model, add it to the config file adopting the following pattern:

        .. code:: yaml

          models:
            DeepFM:
              meta:
                save_recs: True
              epochs: 10
              batch_size: 512
              factors: 100
              lr: 0.001
              l_w: 0.0001
              hidden_neurons: (64,32)
              hidden_activations: ('relu','relu')
        """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_hidden_neurons", "hidden_neurons", "hidden_neurons", "(64,32)", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_hidden_activations", "hidden_activations", "hidden_activations", "('relu','relu')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "reg", "reg", 0.0001, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._sampler = pws.Sampler(self._data.i_train_dict)

        self._model = DeepFMModel(self._num_users,
                                  self._num_items,
                                  self._factors,
                                  tuple(m for m in zip(self._hidden_neurons, self._hidden_activations)),
                                  self._l_w,
                                  self._learning_rate,
                                  random_seed=self._seed)

    @property
    def name(self):
        return "DeepFM" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def predict(self, u: int, i: int):
        pass

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.get_recs(
                (
                    np.repeat(np.array(list(range(offset, offset_stop)))[:, None], repeats=self._num_items, axis=1),
                    np.array([self._i_items_set for _ in range(offset, offset_stop)])
                )
            )
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)

            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

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