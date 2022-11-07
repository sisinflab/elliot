from ast import literal_eval as make_tuple

import pandas as pd
from tqdm import tqdm
import numpy as np
import torch
import os

from elliot.utils.write import store_recommendation
from .pointwise_pos_neg_sampler import Sampler
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .GCMCModel import GCMCModel

from torch_sparse import SparseTensor


class GCMC(RecMixin, BaseRecommenderModel):
    r"""
    Graph Convolutional Matrix Completion

    For further details, please refer to the `paper <https://arxiv.org/abs/1706.02263>`_

    Args:
        lr: Learning rate
        epochs: Number of epochs
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        convolutional_layer_size: Tuple with number of units for each convolutional layer
        dense_layer_size: Tuple with number of units for each dense layer
        node_dropout: Tuple with dropout rate for each node
        dense_layer_dropout: Tuple with hidden layer dropout rate for each dense propagation layer

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        GCMC:
          meta:
            save_recs: True
          lr: 0.0005
          epochs: 50
          batch_size: 512
          factors: 64
          l_w: 0.1
          convolutional_layer_size: (64,)
          dense_layer_size: (64,)
          node_dropout: ()
          dense_layer_dropout: (0.1,)
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_batch_eval", "batch_eval", "batch_eval", 256, int, None),
            ("_convolutional_layer_size", "conv_size", "conv_size", 64, int, None),
            ("_dense_layer_size", "dense_size", "dense_size", 64, int, None),
            ("_n_convolutional_layers", "n_conv", "n_conv", 1, int, None),
            ("_n_dense_layers", "n_dense", "_n_dense", 1, int, None),
            ("_num_rel", "num_rel", "num_rel", 5, int, None),
            ("_num_basis", "num_basis", "num_basis", 2, int, None),
            ("_acc", "acc", "acc", 'stack', str, None)
        ]
        self.autoset_params()

        np.random.seed(self._seed)

        self._sampler = Sampler(self._data.i_train_dict)

        self.df_val_rat = pd.DataFrame(columns=['user', 'item', 'rating'])
        self.df_test_rat = pd.DataFrame(columns=['user', 'item', 'rating'])

        # create dataframes for validation and test set with user-item-rating
        idx = 0
        for k, v in data.val_dict.items():
            for kk, vv in v.items():
                self.df_val_rat = pd.concat([self.df_val_rat,
                                             pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        idx = 0
        for k, v in data.test_dict.items():
            for kk, vv in v.items():
                self.df_test_rat = pd.concat([self.df_test_rat,
                                             pd.DataFrame({'user': k, 'item': int(kk), 'rating': vv}, index=[idx])])
                idx += 1

        self.df_val_rat = self.df_val_rat.astype({'user': int, 'item': int, 'rating': float})
        self.df_test_rat = self.df_test_rat.astype({'user': int, 'item': int, 'rating': float})

        # remove interactions whose user and/or item is not in the training set
        self.df_val_rat = self.df_val_rat[self.df_val_rat['user'] <= self._num_users - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['user'] <= self._num_users - 1]
        self.df_val_rat = self.df_val_rat[self.df_val_rat['item'] <= self._num_items - 1]
        self.df_test_rat = self.df_test_rat[self.df_test_rat['item'] <= self._num_items - 1]

        row, col = self._data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        ratings = self._data.sp_i_train_ratings.data
        edge_index = np.array([row, col, ratings])
        self.adj_ratings = []
        for r in range(1, self._num_rel + 1):
            indices = edge_index[2, :] == r
            edge_index_0 = torch.tensor(edge_index[0, indices], dtype=torch.int64)
            edge_index_1 = torch.tensor(edge_index[1, indices], dtype=torch.int64)
            self.adj_ratings.append(SparseTensor(row=torch.cat([edge_index_0, edge_index_1], dim=0),
                                                 col=torch.cat([edge_index_1, edge_index_0], dim=0),
                                                 sparse_sizes=(self._num_users + self._num_items,
                                                               self._num_users + self._num_items)))

        self._model = GCMCModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            convolutional_layer_size=self._convolutional_layer_size,
            dense_layer_size=self._dense_layer_size,
            n_convolutional_layers=self._n_convolutional_layers,
            n_dense_layers=self._n_dense_layers,
            num_relations=self._num_rel,
            num_basis=self._num_basis,
            adj_ratings=self.adj_ratings,
            accumulation=self._acc,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "GCMC" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        row, col = self._data.sp_i_train.nonzero()
        ratings = self._data.sp_i_train_ratings.data
        edge_index = np.array([row, col, ratings]).transpose()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            np.random.shuffle(edge_index)
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index, self._data.transactions, self._batch_size):
                    steps += 1
                    current_loss = self._model.train_step(batch)
                    loss += current_loss
                    t.set_postfix({'loss': f'{current_loss:.5f}'})
                    t.update()

            self.evaluate(it, loss / steps)

    # def get_recommendations(self, k: int = 100):
    #     predictions_top_k_test = {}
    #     predictions_top_k_val = {}
    #     zu, zi = self._model.propagate_embeddings(evaluate=True)
    #     self.logger.info('Starting predictions on all users/items pairs...')
    #     with tqdm(total=int(self._num_users // self._batch_eval), disable=not self._verbose) as t:
    #         for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
    #             offset_stop = min(offset + self._batch_eval, self._num_users)
    #             predictions = np.empty((offset_stop - offset, self._num_items))
    #             for item_index, item_offset in enumerate(range(0, self._num_items, self._batch_eval)):
    #                 item_offset_stop = min(item_offset + self._batch_eval, self._num_items)
    #                 user_range = np.repeat(np.arange(offset, offset_stop), repeats=item_offset_stop - item_offset)
    #                 item_range = np.tile(np.arange(item_offset, item_offset_stop), reps=offset_stop - offset)
    #                 p = self._model.predict(zu[user_range],
    #                                         zi[item_range],
    #                                         offset_stop - offset,
    #                                         item_offset_stop - item_offset)
    #                 predictions[:, item_offset: item_offset_stop] = p.detach().cpu().numpy()
    #             recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
    #             predictions_top_k_val.update(recs_val)
    #             predictions_top_k_test.update(recs_test)
    #             t.update()
    #     self.logger.info('Predictions on all users/items pairs is complete!')
    #     return predictions_top_k_val, predictions_top_k_test

    def get_recommendations(self, k: int = 100):
        predictions_test = []
        predictions_val = []
        zu, zi = self._model.propagate_embeddings(evaluate=True)
        val_len = len(self.df_val_rat)
        with tqdm(total=int(val_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, val_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, val_len)
                current_df = self.df_val_rat[offset:offset_stop]
                p = self._model.predict(zu[current_df['user'].tolist()],
                                        zi[current_df['item'].tolist()])
                predictions_val += p.detach().cpu().numpy().tolist()
                t.update()
        test_len = len(self.df_test_rat)
        with tqdm(total=int(test_len // self._batch_eval), disable=not self._verbose) as t:
            for index, offset in enumerate(range(0, test_len, self._batch_eval)):
                offset_stop = min(offset + self._batch_eval, test_len)
                current_df = self.df_test_rat[offset:offset_stop]
                p = self._model.predict(zu[current_df['user'].tolist()],
                                        zi[current_df['item'].tolist()])
                predictions_test += p.detach().cpu().numpy().tolist()
                t.update()
        return predictions_val, predictions_test

    def get_single_recommendation(self, mask, k, predictions, offset, offset_stop):
        v, i = self._model.get_top_k(predictions, mask[offset: offset_stop], k=k)
        items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                              for u_list in list(zip(i.detach().cpu().numpy(), v.detach().cpu().numpy()))]
        return dict(zip(map(self._data.private_users.get, range(offset, offset_stop)), items_ratings_pair))

    def evaluate(self, it=None, loss=0.0):
        if (it is None) or (not (it + 1) % self._validation_rate):
            predictions_val, predictions_test = self.get_recommendations()
            true_val, true_test = self.df_val_rat['rating'].to_numpy(), self.df_test_rat['rating'].to_numpy()
            result_dict = self.evaluator.eval_error(np.array(predictions_val), true_val, np.array(predictions_test),
                                                    true_test)

            self._losses.append(loss)

            self._results.append(result_dict)

            if it is not None:
                self.logger.info(f'Epoch {(it + 1)}/{self._epochs} loss {loss:.5f}')
            else:
                self.logger.info(f'Finished')

            if (len(self._results) - 1) == self.get_best_arg():
                if it is not None:
                    self._params.best_iteration = it + 1
                self.logger.info("******************************************")
                self.best_metric_value = self._results[-1][0]["val_results"][self._validation_metric]
                if self._save_weights:
                    if hasattr(self, "_model"):
                        torch.save({
                            'model_state_dict': self._model.state_dict(),
                            'optimizer_state_dict': self._model.optimizer.state_dict()
                        }, self._saving_filepath)
                    else:
                        self.logger.warning("Saving weights FAILED. No model to save.")

    def get_loss(self):
        if self._optimize_internal_loss:
            return min(self._losses)
        else:
            return min([r[0]["val_results"][self._validation_metric] for r in self._results])

    def get_best_arg(self):
        if self._optimize_internal_loss:
            val_results = np.argmin(self._losses)
        else:
            val_results = np.argmin([r[0]["val_results"][self._validation_metric] for r in self._results])
        return val_results

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
