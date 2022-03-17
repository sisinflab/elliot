from ast import literal_eval as make_tuple

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
from .SLATTICEModel import SLATTICEModel


class SLATTICE(RecMixin, BaseRecommenderModel):
    r"""
    Args:
        lr: Learning rate
        epochs: Number of epochs
        n_layers: Number of propagation layers for the item-item graph
        n_ui_layers: Number of propagation layers for the user-item graph
        factors: Number of latent factors
        factors_multimod: Tuple with number of units for each modality
        batch_size: Batch size
        l_w: Regularization coefficient
        int_mod: Tuple of modalities for interactions
        top_k: Top-k for similarity matrix

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        LATTICE:
          meta:
            save_recs: True
          lr: 0.0001
          epochs: 400
          n_layers: 1
          n_ui_layers: 3
          factors: 64
          factors_multimod: 64
          batch_size: 1024
          l_w: 0.000001
          int_mod: (textual,)
          top_k: 100
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
            ("_n_ui_layers", "n_ui_layers", "n_ui_layers", 3, int, None),
            ("_top_k", "top_k", "top_k", 100, int, None),
            ("_factors_multimod", "factors_multimod", "factors_multimod", 64, int, None),
            ("_int_mod", "int_mod", "int_mod", "('textual',)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_int_loads", "int_loads", "int_loads",
             "('SentimentInteractionsTextualAttributes',)",
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]
        self.autoset_params()

        for m_id, m in enumerate(self._int_mod):
            self.__setattr__(f'''_side_{m}''',
                             self._data.side_information.__getattribute__(f'''{self._int_loads[m_id]}'''))

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))
        sim_multimodal = []
        for m in self._int_mod:
            values, rows, cols = self.__getattribute__(f'''_side_{m}''').object.get_all_features(
                self._data.public_items)
            sim_multimodal.append(SparseTensor(row=torch.tensor(np.array(rows), dtype=torch.int64),
                                               col=torch.tensor(np.array(cols), dtype=torch.int64),
                                               value=torch.tensor(np.array(values), dtype=torch.float32)))

        self._model = SLATTICEModel(
            num_users=self._num_users,
            num_items=self._num_items,
            num_layers=self._n_layers,
            num_ui_layers=self._n_ui_layers,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            embed_k_multimod=self._factors_multimod,
            l_w=self._l_w,
            interaction_modalities=self._int_mod,
            top_k=self._top_k,
            sim_multimodal=sim_multimodal,
            adj=self.adj,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "SLATTICE" \
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
                self._model.lr_scheduler.step()

            self.evaluate(it, loss / (it + 1))

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        self._model.eval()
        with torch.no_grad():
            gum, gim = self._model.propagate_embeddings()
            for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
                offset_stop = min(offset + self._batch_size, self._num_users)
                predictions = self._model.predict(gum[offset: offset_stop], gim)
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
