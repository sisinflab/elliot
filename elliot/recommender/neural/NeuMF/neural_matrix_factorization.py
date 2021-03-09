"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from ast import literal_eval as make_tuple
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.neural.NeuMF.neural_matrix_factorization_model import NeuralMatrixFactorizationModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(42)


class NeuMF(RecMixin, BaseRecommenderModel):
    r"""
    Neural Collaborative Filtering

    For further details, please refer to the `paper <https://arxiv.org/abs/1708.05031>`_

    Args:
        mf_factors: Number of MF latent factors
        mlp_factors: Number of MLP latent factors
        mlp_hidden_size: List of units for each layer
        lr: Learning rate
        dropout: Dropout rate
        is_mf_train: Whether to train the MF embeddings
        is_mlp_train: Whether to train the MLP layers

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        NeuMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          mf_factors: 10
          mlp_factors: 10
          mlp_hidden_size: (64,32)
          lr: 0.001
          dropout: 0.0
          is_mf_train: True
          is_mlp_train: True
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._random = np.random

        self._sampler = pws.Sampler(self._data.i_train_dict)

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_mf_factors", "mf_factors", "mffactors", 10, int, None),
            ("_mlp_factors", "mlp_factors", "mlpfactors", 10, int, None),
            ("_mlp_hidden_size", "mlp_hidden_size", "mlpunits", "(64,32)", lambda x: list(make_tuple(str(x))), lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout", "dropout", "drop", 0, None, None),
            ("_is_mf_train", "is_mf_train", "mftrain", True, None, None),
            ("_is_mlp_train", "is_mlp_train", "mlptrain", True, None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = NeuralMatrixFactorizationModel(self._num_users, self._num_items, self._mf_factors,
                                                     self._mlp_factors, self._mlp_hidden_size,
                                                     self._dropout, self._is_mf_train, self._is_mlp_train,
                                                     self._learning_rate)

    @property
    def name(self):
        return "NeuMF"\
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
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss/steps:.5f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.get_recs(
                (
                    np.repeat(np.array(list(range(offset, offset_stop)))[:, None], repeats=self._num_items, axis=1),
                    np.array([self._i_items_set for _ in range(offset, offset_stop)])
                 )
            )
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(map(self._data.private_users.get,
                                                  range(offset, offset_stop)), items_ratings_pair)))
        return predictions_top_k
