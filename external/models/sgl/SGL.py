from tqdm import tqdm
import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .SGLModel import SGLModel

from torch_sparse import SparseTensor

import random


class SGL(RecMixin, BaseRecommenderModel):
    r"""
    Self-supervised Graph Learning for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3404835.3462862>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        n_layers: Number of stacked propagation layers
        ssl_temp: Temperature
        ssl_reg: Regularization term for ssl
        ssl_ratio: Dropout ratio for ssl
        sampling: Sampling strategy

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        SGL:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          batch_size: 256
          l_w: 0.1
          n_layers: 2
          ssl_temp: 0.2
          ssl_reg: 0.1
          ssl_ratio: 0.1
          sampling: nd
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
            ("_l_w", "l_w", "l_w", 0.01, float, None),
            ("_n_layers", "n_layers", "n_layers", 1, int, None),
            ("_ssl_temp", "ssl_temp", "ssl_temp", 0.2, float, None),
            ("_ssl_reg", "ssl_reg", "ssl_reg", 0.1, float, None),
            ("_ssl_ratio", "ssl_ratio", "ssl_ratio", 0.1, float, None),
            ("_sampling", "sampling", "sampling", 'nd', str, None)
        ]
        self.autoset_params()

        random.seed(self._seed)
        np.random.seed(self._seed)

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        self.edge_index = np.array([row, col])
        adj = SparseTensor(row=torch.cat([torch.tensor(self.edge_index[0], dtype=torch.int64),
                                          torch.tensor(self.edge_index[1], dtype=torch.int64)], dim=0),
                           col=torch.cat([torch.tensor(self.edge_index[1], dtype=torch.int64),
                                          torch.tensor(self.edge_index[0], dtype=torch.int64)], dim=0),
                           sparse_sizes=(self._num_users + self._num_items,
                                         self._num_users + self._num_items))
        self.users = list(range(self._num_users))
        self.items = list(range(self._num_items))
        self.interactions = list(range(self.edge_index.shape[1]))

        self._model = SGLModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            n_layers=self._n_layers,
            ssl_temp=self._ssl_temp,
            ssl_reg=self._ssl_reg,
            sampling=self._sampling,
            adj=adj,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "SGL" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            if self._sampling in ['nd', 'ed']:
                adj_1 = self.create_adj_mat(is_subgraph=True, aug_type=self._sampling)
                adj_2 = self.create_adj_mat(is_subgraph=True, aug_type=self._sampling)
            elif self._sampling == 'rw':
                adj_1, adj_2 = [], []
                for _ in range(self._n_layers):
                    adj_1.append(self.create_adj_mat(is_subgraph=True, aug_type=self._sampling))
                    adj_2.append(self.create_adj_mat(is_subgraph=True, aug_type=self._sampling))
            else:
                raise NotImplementedError('This sampling strategy has not been implemented yet!')
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch, adj_1, adj_2)
                    t.set_postfix({'loss': f'{loss / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss / (it + 1))

    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        if is_subgraph and self._ssl_ratio > 0:
            if aug_type == 'nd':
                users_to_drop = random.sample(self.users, round(self._num_users * self._ssl_ratio))
                items_to_drop = random.sample(self.items, round(self._num_items * self._ssl_ratio))
                mask_user = ~np.isin(self.edge_index[0], list(users_to_drop))
                mask_item = ~np.isin(self.edge_index[1], list(items_to_drop))
                sampled_edge_index = self.edge_index[:, mask_user & mask_item]
                sampled_edge_index = torch.tensor(sampled_edge_index, dtype=torch.int64)
                sampled_adj = SparseTensor(row=torch.cat([sampled_edge_index[0], sampled_edge_index[1]], dim=0),
                                           col=torch.cat([sampled_edge_index[1], sampled_edge_index[0]], dim=0),
                                           sparse_sizes=(self._num_users + self._num_items,
                                                         self._num_users + self._num_items))
            elif aug_type in ['ed', 'rw']:
                interactions_to_drop = random.sample(self.interactions,
                                                     round(self.edge_index.shape[1] * self._ssl_ratio))
                mask_interactions = ~np.isin(np.array(self.interactions), list(interactions_to_drop))
                sampled_edge_index = self.edge_index[:, mask_interactions]
                sampled_edge_index = torch.tensor(sampled_edge_index, dtype=torch.int64)
                sampled_adj = SparseTensor(row=torch.cat([sampled_edge_index[0], sampled_edge_index[1]], dim=0),
                                           col=torch.cat([sampled_edge_index[1], sampled_edge_index[0]], dim=0),
                                           sparse_sizes=(self._num_users + self._num_items,
                                                         self._num_users + self._num_items))

            else:
                raise NotImplementedError('This sampling strategy has not been implemented yet!')
        else:
            edge_index = torch.tensor(self.edge_index, dtype=torch.int64)
            return SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))
        return sampled_adj

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
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
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss / (it + 1):.5f}')
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
                self.best_metric_value = self._results[-1][self._validation_k]["val_results"][
                    self._validation_metric]
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
