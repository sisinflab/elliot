"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import numpy as np
from tqdm import tqdm

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.adversarial.AMF.AMF_model import AMF_model
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from elliot.utils.write import store_recommendation


class AMF(RecMixin, BaseRecommenderModel):
    r"""
    Adversarial Matrix Factorization

    For further details, please refer to the `paper <https://arxiv.org/abs/1808.03908>`_

    Args:
        factors: Number of latent factor
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        eps: Perturbation Budget
        l_adv: Adversarial regularization coefficient
        adversarial_epochs: Adversarial epochs

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        AMF:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 200
          lr: 0.001
          l_w: 0.1
          l_b: 0.001
          eps: 0.1
          l_adv: 0.001
          adversarial_epochs: 10

    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a APR-MF (AMF) instance.
        (see https://arxiv.org/abs/1808.03908 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      [eps, l_adv]: adversarial budget perturbation and adversarial regularization parameter,
                                      lr: learning rate}
        """

        self._params_list = [
            ("_factors", "factors", "factors", 200, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.1, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_eps", "eps", "eps", 0.1, None, None),
            ("_l_adv", "l_adv", "l_adv", 0.001, None, None),
            ("_adversarial_epochs", "adversarial_epochs", "adv_epochs", self._epochs//2, int, None)
        ]

        self.autoset_params()

        if self._adversarial_epochs > self._epochs:
            raise Exception(f"The total epoch ({self._epochs}) "
                            f"is smaller than the adversarial epochs ({self._adversarial_epochs}).")

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = cs.Sampler(self._data.i_train_dict)

        self._model = AMF_model(self._factors,
                                self._learning_rate,
                                self._l_w,
                                self._l_b,
                                self._eps,
                                self._l_adv,
                                self._num_users,
                                self._num_items,
                                self._seed)

    @property
    def name(self):
        return "AMF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            user_adv_train = False if it < self._adversarial_epochs else True
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch, user_adv_train)
                    # t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                    t.set_postfix({'(APR)-loss' if user_adv_train else '(BPR)-loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy()/(it + 1))

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