"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import tensorflow as tf
import numpy as np

from tqdm import tqdm

from elliot.recommender.adversarial.AMR import pairwise_pipeline_sampler_vbpr as ppsv
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.adversarial.AMR.AMR_model import AMR_model
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin


class AMR(RecMixin, BaseRecommenderModel):
    r"""
    Adversarial Multimedia Recommender

    For further details, please refer to the `paper <https://arxiv.org/pdf/1809.07062.pdf>`_

    Args:
        factors: Number of latent factor
        factors_d: Image-feature dimensionality
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        l_e: Regularization coefficient of image matrix embedding
        eps: Perturbation Budget
        l_adv: Adversarial regularization coefficient
        adversarial_epochs: Adversarial epochs

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        AMR:
          meta:
            save_recs: True
          epochs: 10
          batch_size: 512
          factors: 200
          factors_d: 20
          lr: 0.001
          l_w: 0.1
          l_b: 0.001
          l_e: 0.1
          eps: 0.1
          l_adv: 0.001
          adversarial_epochs: 5
    """
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a AMR instance.
        (see https://arxiv.org/pdf/1809.07062.pdf for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      [eps, l_adv]: adversarial budget perturbation and adversarial regularization parameter,
                                      lr: learning rate}
        """
        super().__init__(data, config, params, *args, **kwargs)

        self._num_items = self._data.num_items
        self._num_users = self._data.num_users

        self._params_list = [
            ("_batch_eval", "batch_eval", "be", 512, int, None),
            ("_factors", "factors", "factors", 200, int, None),
            ("_factors_d", "factors_d", "factors_d", 20, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_l_w", "l_w", "l_w", 0.1, None, None),
            ("_l_b", "l_b", "l_b", 0.001, None, None),
            ("_l_e", "l_e", "l_e", 0.1, None, None),
            ("_eps", "eps", "eps", 0.1, None, None),
            ("_l_adv", "l_adv", "l_adv", 0.001, None, None),
            ("_adversarial_epochs", "adversarial_epochs", "adv_epochs", self._epochs // 2, int, None),
            ("_loader", "loader", "load", "VisualAttributes", None, None)
        ]
        self.autoset_params()

        if self._adversarial_epochs > self._epochs:
            raise Exception(f"The total epoch ({self._epochs}) "
                            f"is smaller than the adversarial epochs ({self._adversarial_epochs}).")

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._side = getattr(self._data.side_information, self._loader, None)

        item_indices = [self._side.item_mapping[self._data.private_items[item]] for item in range(self._num_items)]

        self._sampler = ppsv.Sampler(self._data.i_train_dict,
                                     item_indices,
                                     self._side.visual_feature_folder_path,
                                     self._epochs)

        self._next_batch = self._sampler.pipeline(self._data.transactions, self._batch_size)

        self._model = AMR_model(self._factors,
                                self._factors_d,
                                self._learning_rate,
                                self._l_w,
                                self._l_b,
                                self._l_e,
                                self._side.visual_features_shape,
                                self._num_users,
                                self._num_items,
                                self._eps,
                                self._l_adv,
                                self._seed)

        self._next_eval_batch = self._sampler.pipeline_eval(self._batch_eval)

    @property
    def name(self):
        return "AMR" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        loss = 0
        steps = 0
        it = 0
        user_adv_train = (self._epochs - it) <= self._adversarial_epochs
        with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
            for batch in self._next_batch:
                steps += 1
                loss += self._model.train_step(batch, user_adv_train)
                t.set_postfix({'(APR)-loss' if user_adv_train else '(BPR)-loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

                if steps == self._data.transactions // self._batch_size:
                    t.reset()
                    self.evaluate(it, loss.numpy() / steps)
                    it += 1
                    steps = 0
                    loss = 0
                    user_adv_train = (self._epochs - it) <= self._adversarial_epochs

    def get_recommendations(self, k: int = 100):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_eval)):
            offset_stop = min(offset + self._batch_eval, self._num_users)
            predictions = np.empty((offset_stop - offset, self._num_items))
            for batch in self._next_eval_batch:
                item_rel, item_abs, feat = batch
                p = self._model.predict_item_batch(offset, offset_stop,
                                                   item_rel[0], item_rel[-1],
                                                   tf.Variable(feat))
                predictions[:(offset_stop - offset), item_rel] = p
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

