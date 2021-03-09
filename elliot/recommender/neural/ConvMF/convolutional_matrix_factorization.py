"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from ast import literal_eval as make_tuple
from tqdm import tqdm
import pickle

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender.neural.ConvMF.convolutional_matrix_factorization_model import \
    ConvMatrixFactorizationModel
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

from elliot.recommender.base_recommender_model import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger

np.random.seed(42)


class ConvMF(RecMixin, BaseRecommenderModel):
    r"""
        Convolutional Matrix Factorization for Document Context-Aware Recommendation

        For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/2959100.2959165>`_

        Args:
            embedding_size: Embedding dimension
            lr: Learning rate
            l_w: Regularization coefficient
            l_b: Regularization coefficient of bias
            cnn_channels: List of channels
            cnn_kernels: List of kernels
            cnn_strides: List of strides
            dropout_prob: Dropout probability applied on the convolutional layers

        To include the recommendation model, add it to the config file adopting the following pattern:

        .. code:: yaml

          models:
            ConvMF:
              meta:
                save_recs: True
              epochs: 10
              batch_size: 512
              embedding_size: 100
              lr: 0.001
              l_w: 0.005
              l_b: 0.0005
              cnn_channels: (1, 32, 32)
              cnn_kernels: (2,2)
              cnn_strides: (2,2)
              dropout_prob: 0
        """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Args:
            data:
            config:
            params:
            *args:
            **kwargs:
        """
        self._random = np.random

        self._sampler = pws.Sampler(self._data.i_train_dict)

        self._params_list = [
            ("_lr", "lr", "lr", 0.001, None, None),
            ("_embedding_size", "embedding_size", "embedding_size", 100, None, None),
            ("_cnn_channels", "cnn_channels", "cnn_channels", "(1, 32, 32)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_cnn_kernels", "cnn_kernels", "cnn_kernels", "(2,2)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_cnn_strides", "cnn_strides", "cnn_strides", "(2,2)", lambda x: list(make_tuple(str(x))),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_dropout_prob", "dropout_prob", "dropout_prob", 0, None, None),
            ("_l_w", "l_w", "l_w", 0.005, None, None),
            ("_l_b", "l_b", "l_b", 0.0005, None, None),
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict
        self._sp_i_train = self._data.sp_i_train
        self._i_items_set = list(range(self._num_items))

        self._model = ConvMatrixFactorizationModel(self._num_users, self._num_items, self._embedding_size,
                                                         self._lr, self._cnn_channels, self._cnn_kernels,
                                                         self._cnn_strides, self._dropout_prob, self._l_w, self._l_b
                                                         )

    @property
    def name(self):
        return "ConvMF" \
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

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss / steps:.5f}')

                if self._results[-1][self._validation_k]["val_results"][
                    self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][
                        self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs,
                                             self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.get_recs(
                (np.repeat(np.array(list(range(offset, offset_stop)))[:, None], repeats=self._num_items, axis=1),
                 np.array([self._i_items_set for _ in range(offset, offset_stop)])))
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(map(self._data.private_users.get,
                                                  range(offset, offset_stop)), items_ratings_pair)))
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
