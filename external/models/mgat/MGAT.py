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
from elliot.dataset.samplers import custom_sampler_full as csf
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .MGATModel import MGATModel

from torch_sparse import SparseTensor


class MGAT(RecMixin, BaseRecommenderModel):
    r"""
    MGAT: Multimodal Graph Attention Network for Recommendation

    For further details, please refer to the `paper <https://www.sciencedirect.com/science/article/abs/pii/S0306457320300182?via%3Dihub>`_

    Args:
        lr: Learning rate
        l_w: Regularizer
        epochs: Number of epochs
        num_layers: Number of propagation layers
        num_routings: Number of routing iterations
        factors: Number of latent factors
        factors_multimod: Tuple with number of units for each modality
        batch_size: Batch size
        modalities: Tuple of modalities

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        MGAT:
          meta:
            save_recs: True
          lr: 0.0005
          l_w: 0.0001
          epochs: 50
          num_layers: 3
          num_routings: 10
          factors: 64
          factors_multimod: (64,64)
          batch_size: 256
          modalities: (visual,textual)
    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        ######################################

        self._params_list = [
            ("_learning_rate", "lr", "lr", 0.0005, float, None),
            ("_l_w", "l_w", "l_w", 0.0005, float, None),
            ("_factors", "factors", "factors", 64, int, None),
            ("_num_layers", "num_layers", "num_layers", 3, int, None),
            ("_factors_multimod", "factors_multimod", "factors_multimod", (256, None),
             lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_modalities", "modalities", "modalites", "('visual','textual')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-")),
            ("_loaders", "loaders", "loads", "('VisualAttribute','TextualAttribute')", lambda x: list(make_tuple(x)),
             lambda x: self._batch_remove(str(x), " []").replace(",", "-"))
        ]
        self.autoset_params()

        np.random.seed(self._seed)

        self._sampler = csf.Sampler(self._data.i_train_dict, self._seed)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        for m_id, m in enumerate(self._modalities):
            self.__setattr__(f'''_side_{m}''',
                             self._data.side_information.__getattribute__(f'''{self._loaders[m_id]}'''))

        row, col = data.sp_i_train.nonzero()
        col = [c + self._num_users for c in col]
        _, counts_full = np.unique(np.concatenate((row, np.array(col))), return_counts=True)
        ptr_full = [0]
        c_full = counts_full.tolist()
        for i in range(len(c_full)):
            ptr_full += [ptr_full[i] + c_full[i]]
        ptr_full = torch.tensor(np.array(ptr_full), dtype=torch.int64)
        edge_index = np.array([row, col])
        edge_index = torch.tensor(edge_index, dtype=torch.int64)
        self.adj = SparseTensor(row=torch.cat([edge_index[0], edge_index[1]], dim=0),
                                col=torch.cat([edge_index[1], edge_index[0]], dim=0),
                                sparse_sizes=(self._num_users + self._num_items,
                                              self._num_users + self._num_items))

        self._model = MGATModel(
            num_users=self._num_users,
            num_items=self._num_items,
            learning_rate=self._learning_rate,
            embed_k=self._factors,
            l_w=self._l_w,
            embed_k_multimod=self._factors_multimod,
            num_layers=self._num_layers,
            modalities=self._modalities,
            multimodal_features=[self.__getattribute__(f'''_side_{m}''').object.get_all_features() for m in
                                 self._modalities],
            adj=self.adj,
            rows=row,
            cols=col,
            ptr=ptr_full,
            random_seed=self._seed
        )

    @property
    def name(self):
        return "MGAT" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        row, col = self._data.sp_i_train.nonzero()
        edge_index = np.array([row, col]).transpose()

        for it in self.iterate(self._epochs):
            loss = 0
            steps = 0
            self._model.train()
            np.random.shuffle(edge_index)
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(edge_index, self._data.transactions, self._batch_size):
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
