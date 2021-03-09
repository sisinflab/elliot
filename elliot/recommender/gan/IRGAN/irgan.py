"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import pointwise_pos_neg_sampler as pws
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.gan.IRGAN.irgan_model import IRGAN_model
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation

np.random.seed(42)


class IRGAN(RecMixin, BaseRecommenderModel):
    r"""
    IRGAN: A Minimax Game for Unifying Generative and Discriminative Information Retrieval Models

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3077136.3080786>`_

    Args:
        factors: Number of latent factor
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        l_gan: Adversarial regularization coefficient
        predict_model: Specification of the model to generate the recommendation (Generator/ Discriminator)
        g_epochs: Number of epochs to train the generator for each IRGAN step
        d_epochs: Number of epochs to train the discriminator for each IRGAN step
        g_pretrain_epochs: Number of epochs to pre-train the generator
        d_pretrain_epochs: Number of epochs to pre-train the discriminator
        sample_lambda: Temperature Parameters

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        IRGAN:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 10
          lr: 0.001
          l_w: 0.1
          l_b: 0.001
          l_gan: 0.001
          predict_model: generator
          g_epochs: 5
          d_epochs: 1
          g_pretrain_epochs: 10
          d_pretrain_epochs: 10
          sample_lambda: 0.2
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):

        self._random = np.random

        self._params_list = [
            ("_predict_model", "predict_model", "predict_model", "generator", None, None),
            ("_factors", "factors", "factors", 10, None, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.1, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_l_gan", "l_gan", "l_gan", 0.001, None, None),
            ("_g_epochs", "g_epochs", "g_epochs", 5, None, None),
            ("_d_epochs", "d_epochs", "d_epochs", 1, None, None),
            ("_g_pretrain_epochs", "g_pretrain_epochs", "g_pt_ep", 10, None, None),
            ("_d_pretrain_epochs", "d_pretrain_epochs", "d_pt_ep", 10, None, None),
            ("_sample_lambda", "sample_lambda", "sample_lambda", 0.2, None, None)
        ]
        self.autoset_params()

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        if self._predict_model not in ["generator", "discriminator"]:
            raise Exception(f"It is necessary to specify the model component to use as recommender (generator/discriminator)")

        self._ratings = self._data.train_dict

        self._sampler = pws.Sampler(self._data.i_train_dict)

        self._model = IRGAN_model(self._predict_model,
                                  self._data,
                                  self._batch_size,
                                  self._factors,
                                  self._learning_rate,
                                  self._l_w,
                                  self._l_b,
                                  self._l_gan,
                                  self._num_users,
                                  self._num_items,
                                  self._g_pretrain_epochs,
                                  self._d_pretrain_epochs,
                                  self._g_epochs,
                                  self._d_epochs,
                                  self._sample_lambda)

    @property
    def name(self):
        return "IRGAN" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        for it in range(self._epochs):
            dis_loss, gen_loss = 0, 0
            with tqdm(total=1, disable=not self._verbose) as t:
                update_dis_loss, update_gen_loss = self._model.train_step()
                dis_loss += update_dis_loss
                gen_loss += update_gen_loss
                t.set_postfix(
                    {'Dis loss': f'{dis_loss.numpy():.5f}', 'Gen loss': f'{gen_loss.numpy():.5f}'})
                t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} Dis loss: {dis_loss.numpy():.5f}, Gen loss: {gen_loss.numpy():.5f}')

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
            offset_stop = min(offset + self._params.batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop)
            mask = self.get_train_mask(offset, offset_stop)
            v, i = self._model.get_top_k(predictions, mask, k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(range(offset, offset_stop), items_ratings_pair)))
        return predictions_top_k
