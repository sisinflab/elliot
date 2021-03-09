"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'
__paper__ = 'FISM: Factored Item Similarity Models for Top-N Recommender Systems by Santosh Kabbur, Xia Ning, and George Karypis'

import numpy as np
from tqdm import tqdm
import pickle

from elliot.dataset.samplers import pointwise_pos_neg_ratio_ratings_sampler as pws
from elliot.utils.write import store_recommendation

from elliot.recommender import BaseRecommenderModel
from elliot.recommender.latent_factor_models.FISM.FISM_model import FISM_model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(42)


class FISM(RecMixin, BaseRecommenderModel):
    r"""
    FISM: Factored Item Similarity Models

    For further details, please refer to the `paper <http://glaros.dtc.umn.edu/gkhome/node/1068>`_

    Args:
        factors: Number of factors of feature embeddings
        lr: Learning rate
        l_w: Regularization coefficient for latent factors
        l_b: Regularization coefficient for bias
        alpha: Alpha parameter (a value between 0 and 1)
        neg_ratio:

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        FISM:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.001
          l_b: 0.001
          alpha: 0.5
          neg_ratio:
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """

        Create a FISM instance.
        (see http://glaros.dtc.umn.edu/gkhome/node/1068 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {_factors: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_lr", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_alpha", "alpha", "alpha", 0.5, lambda x: min(max(0, x), 1), None),
            ("_neg_ratio", "neg_ratio", "neg_ratio", 0.5, None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = pws.Sampler(self._data.i_train_dict, self._data.sp_i_train_ratings, self._neg_ratio)

        self._model = FISM_model(self._data,
                                self._factors,
                                self._lr,
                                self._l_w,
                                self._l_b,
                                self._alpha,
                                self._num_users,
                                self._num_items)

    @property
    def name(self):
        return "FISM" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        for it in range(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch)
                    t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                print(f'Epoch {(it + 1)}/{self._epochs} Get recommendations')
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss / steps:.3f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100, auc_compute: bool = False):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.batch_predict(offset, offset_stop)
            # predictions = self._model.predict(offset) # Single User
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k

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

