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

np.random.seed(42)


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
        self._random = np.random

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
                                    self._num_items)

    @property
    def name(self):
        return "AMF" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        for it in range(self._epochs):
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

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss  / steps:.3f}')

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