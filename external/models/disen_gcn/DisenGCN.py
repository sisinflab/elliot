"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from ast import literal_eval as make_tuple

from tqdm import tqdm
import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .DisenGCNModel import DisenGCNModel
from elliot.dataset.samplers import custom_sampler as cs


class DisenGCN(RecMixin, BaseRecommenderModel):
    r"""
    Disentangled Graph Convolutional Networks

    For further details, please refer to the `paper <http://proceedings.mlr.press/v97/ma19a.html>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        weight_size: Tuple with number of units for each embedding propagation layer
        message_dropout: Dropout rate for each embedding propagation layer
        disen_k: Tuple with factor for disentanglement for each embedding propagation layer
        temperature: Temperature value for softmax
        routing_iterations: Number of routing iterations

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        DisenGCN:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          factors: 64
          batch_size: 512
          l_w: 0.1
          weight_size: (64,)
          message_dropout: 0.1
          disen_k: (10,)
          temperature: 10
          routing_iterations: 5
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        self._sampler = cs.Sampler(self._data.i_train_dict)
        if self._batch_size < 1:
            self._batch_size = self._num_users
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_n_layers", "n_layers", "n_layers", 3, int, None),
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_message_dropout", "message_dropout", "message_dropout", 0.1, float, None),
            ("_disen_k", "disen_k", "disen_k", "()", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_temperature", "temperature", "temperature", 10, float, None),
            ("_routing_iterations", "routing_iterations", "routing_iterations", 5, int, None)
        ]
        self.autoset_params()

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        self.edge_index = np.array([list(row) + col, col + list(row)])

        self._model = DisenGCNModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            n_layers=self._n_layers,
            disen_k=self._disen_k,
            temperature=self._temperature,
            routing_iterations=self._routing_iterations,
            message_dropout=self._message_dropout,
            edge_index=self.edge_index,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "DisenGCN" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

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
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        gu, gi = self._model.propagate_embeddings(evaluate=True)
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(gu[offset: offset_stop], gi)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
            result_dict = self.evaluator.eval(recs)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss/(it + 1):.5f}')
            else:
                self.logger.info(f'Finished')

            if self._save_recs:
                self.logger.info(f"Writing recommendations at: {self._config.path_output_rec_result}")
                if it is not None:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}_it={it + 1}.tsv"])))
                else:
                    store_recommendation(recs[1], os.path.abspath(
                        os.sep.join([self._config.path_output_rec_result, f"{self.name}.tsv"])))

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def restore_weights(self):
        try:
            checkpoint = torch.load(self._saving_filepath)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model correctly Restored")
            self.evaluate()
            return True

        except Exception as ex:
            raise Exception(f"Error in model restoring operation! {ex}")

        return False
