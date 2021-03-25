"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_cfgan_sampler as pwcfgans
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.gan.CFGAN.cfgan_model import CFGAN_model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

np.random.seed(42)


class CFGAN(RecMixin, BaseRecommenderModel):
    r"""
    CFGAN: A Generic Collaborative Filtering Framework based on Generative Adversarial Networks

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3269206.3271743>`_

    Args:
        factors: Number of latent factor
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        l_gan: Adversarial regularization coefficient
        g_epochs: Number of epochs to train the generator for each IRGAN step
        d_epochs: Number of epochs to train the discriminator for each IRGAN step
        s_zr: Sampling parameter of zero-reconstruction
        s_pm: Sampling parameter of partial-masking

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        CFGAN:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.1
          l_b: 0.001
          l_gan: 0.001
          g_epochs: 5
          d_epochs: 1
          s_zr: 0.001
          s_pm: 0.001
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a CFGAN instance.
        (see https://dl.acm.org/doi/10.1145/3269206.3271743 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      lr: learning rate
                                      embed_k: 50
                                      [ l_w, l_b]: regularization
                                      predict_model: generator # or discriminator
                                      s_zr: sampling parameter of zero-reconstruction
                                      s_pm: sampling parameter of partial-masking
                                      l_gan: gan regularization coeff
                                      }
        """
        self._random = np.random

        self._params_list = [
            ("_factors", "factors", "factors", 10, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.1, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_l_gan", "l_gan", "l_gan", 0.001, None, None),
            ("_g_epochs", "g_epochs", "g_epochs", 5, int, None),
            ("_d_epochs", "d_epochs", "d_epochs", 1, int, None),
            ("_s_zr", "s_zr", "s_zr", 0.001, None, None),  # sampling parameter of zero-reconstruction
            ("_s_pm", "s_pm", "s_pm", 0.001, None, None)  # sampling parameter of partial-masking
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = pwcfgans.Sampler(self._data.i_train_dict, self._data.sp_i_train, self._s_zr, self._s_pm)

        self._model = CFGAN_model(self._data,
                                  self._batch_size,
                                  self._learning_rate,
                                  self._l_w,
                                  self._l_b,
                                  self._l_gan,
                                  self._num_users,
                                  self._num_items,
                                  self._g_epochs,
                                  self._d_epochs,
                                  self._s_zr,
                                  self._s_pm
                                  )

    @property
    def name(self):
        return "CFGAN" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0

        for it in range(self._epochs):
            dis_loss, gen_loss = 0, 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    update_dis_loss, update_gen_loss = self._model.train_step(batch)
                    dis_loss += update_dis_loss
                    gen_loss += update_gen_loss
                    t.set_postfix({'Dis loss': f'{dis_loss.numpy() / steps:.5f}', 'Gen loss': f'{gen_loss.numpy() / steps:.5f}'})
                    t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} Dis loss: {dis_loss.numpy() / steps:.5f}, Gen loss: {gen_loss.numpy() / steps:.5f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._params.batch_size)):
            offset_stop = min(offset+self._params.batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k