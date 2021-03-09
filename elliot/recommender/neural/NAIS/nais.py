"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'
__paper__ = 'FISM: Factored Item Similarity Models for Top-N Recommender Systems by Santosh Kabbur, Xia Ning, and George Karypis'


import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_pos_neg_ratio_ratings_sampler as pws
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.neural.NAIS.nais_model import NAIS_model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

np.random.seed(42)


class NAIS(RecMixin, BaseRecommenderModel):
    r"""
    NAIS: Neural Attentive Item Similarity Model for Recommendation

    For further details, please refer to the `paper <https://arxiv.org/abs/1809.07053>`_

    Args:
        factors: Number of latent factors
        algorithm: Type of user-item factor operation ('product', 'concat')
        weight_size: List of units for each layer
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Bias regularization coefficient
        alpha: Attention factor
        beta: Smoothing exponent
        neg_ratio: Ratio of negative sampled items, e.g., 0 = no items, 1 = all un-rated items

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NAIS:
          meta:
            save_recs: True
          factors: 100
          batch_size: 512
          algorithm: concat
          weight_size: 32
          lr: 0.001
          l_w: 0.001
          l_b: 0.001
          alpha: 0.5
          beta: 0.5
          neg_ratio: 0.5
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """

        Create a NAIS instance.
        (see https://arxiv.org/pdf/1809.07053.pdf for details about the algorithm design choices).

        """
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "factors", 100, None, None),
            ("_algorithm", "algorithm", "algorithm", "concat", None, None),
            ("_weight_size", "weight_size", "weight_size", 32, None, None),
            ("_lr", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.001, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_alpha", "alpha", "alpha", 0.5, lambda x: min(max(0, x), 1), None),
            ("_beta", "beta", "beta", 0.5, None, None),
            ("_neg_ratio", "neg_ratio", "neg_ratio", 0.5, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = pws.Sampler(self._data.i_train_dict, self._data.sp_i_train_ratings, self._neg_ratio)

        self._model = NAIS_model(self._data,
                                 self._algorithm,
                                 self._weight_size,
                                 self._factors,
                                 self._lr,
                                 self._l_w,
                                 self._l_b,
                                 self._alpha,
                                 self._beta,
                                 self._num_users,
                                 self._num_items)

    @property
    def name(self):
        return "NAIS" \
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
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k

    # def restore_weights(self):
    #     try:
    #         with open(self._saving_filepath, "rb") as f:
    #             self._model.set_model_state(pickle.load(f))
    #         print(f"Model correctly Restored")
    #
    #         recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
    #         result_dict = self.evaluator.eval(recs)
    #         self._results.append(result_dict)
    #
    #         print("******************************************")
    #         if self._save_recs:
    #             store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}.tsv")
    #         return True
    #
    #     except Exception as ex:
    #         print(f"Error in model restoring operation! {ex}")
    #
    #     return False
