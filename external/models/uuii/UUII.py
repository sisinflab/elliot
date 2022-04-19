from tqdm import tqdm
import torch
import os
import numpy as np

from elliot.utils.write import store_recommendation
from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from torch_sparse import SparseTensor
from .UUIIModel import UUIIModel


class UUII(RecMixin, BaseRecommenderModel):
    r"""
    Args:
        lr: Learning rate
        epochs: Number of epochs
        n_uu_layers: Number of propagation layers for the user-user graph
        n_ii_layers: Number of propagation layers for the item-item graph
        factors: Number of latent factors
        batch_size: Batch size
        l_w: Regularization coefficient
        top_k_uu: Top-k for user-user similarity matrix
        top_k_ii: Top-k for item-item similarity matrix
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
            ("_n_uu_layers", "n_uu_layers", "n_uu_layers", 2, int, None),
            ("_n_ii_layers", "n_ii_layers", "n_ii_layers", 2, int, None),
            ("_top_k_uu", "top_k_uu", "top_k_uu", 100, int, None),
            ("_top_k_ii", "top_k_ii", "top_k_ii", 100, int, None),
            ("_loader", "loader", "loader", 'SentimentInteractionsTextualAttributesUUII', str, None)
        ]
        self.autoset_params()

        self._side_edge_textual = self._data.side_information.SentimentInteractionsTextualAttributesUUII
        all_interactions_uu, all_interactions_ii, rows_uu, rows_ii, cols_uu, cols_ii = self._side_edge_textual.object.get_all_features(
            self._data.public_users, self._data.public_items)

        sim_uu = SparseTensor(row=torch.tensor(np.array(rows_uu), dtype=torch.int64),
                              col=torch.tensor(np.array(cols_uu), dtype=torch.int64),
                              value=torch.tensor(np.array(all_interactions_uu), dtype=torch.float32))
        sim_ii = SparseTensor(row=torch.tensor(np.array(rows_ii), dtype=torch.int64),
                              col=torch.tensor(np.array(cols_ii), dtype=torch.int64),
                              value=torch.tensor(np.array(all_interactions_ii), dtype=torch.float32))

        self._model = UUIIModel(
            num_users=self._num_users,
            num_items=self._num_items,
            num_uu_layers=self._n_uu_layers,
            num_ii_layers=self._n_ii_layers,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            top_k_uu=self._top_k_uu,
            top_k_ii=self._top_k_ii,
            sim_uu=sim_uu,
            sim_ii=sim_ii,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "UUII" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            self._model.train()
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
        self._model.eval()
        with torch.no_grad():
            gu, gi = self._model.propagate_embeddings()
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
