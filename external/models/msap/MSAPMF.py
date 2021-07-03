"""
Module description:

"""

__version__ = '0.3.0'
__author__ = 'Felice Antonio Merra, Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'felice.merra@poliba.it, vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

from tqdm import tqdm
import pandas as pd
import os

from elliot.dataset.samplers import custom_sampler as cs
from elliot.recommender import BaseRecommenderModel
from elliot.recommender.base_recommender_model import init_charger
from elliot.recommender.recommender_utils_mixin import RecMixin
from .MSAPMF_model import MSAPMF_model


class MSAPMF(RecMixin, BaseRecommenderModel):
    r"""
    Adversarial Matrix Factorization

    MSAP presented by Anelli et al. in `paper <https://journals.flvc.org/FLAIRS/article/view/128443>`


    Args:
        meta:
            eval_perturbations: If True Elliot evaluates the effects of both FGSM and MSAP perturbations for each validation epoch
        factors: Number of latent factor
        lr: Learning rate
        l_w: Regularization coefficient
        l_b: Regularization coefficient of bias
        eps: Perturbation Budget
        l_adv: Adversarial regularization coefficient
        adversarial_epochs: Adversarial epochs
        eps_iter: Size of perturbations in MSAP perturbations
        nb_iter: Number of Iterations in MSAP perturbations

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        AMF:
          meta:
            save_recs: True
            eval_perturbations: True
          epochs: 10
          batch_size: 512
          factors: 200
          lr: 0.001
          l_w: 0.1
          l_b: 0.001
          eps: 0.1
          l_adv: 0.001
          adversarial_epochs: 10
          nb_iter: 20
          eps_iter: 0.00001  # If not specified = 2.5*eps/nb_iter

    """

    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """


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
            ("_eps_iter", "eps_iter", "eps_iter", None, None, None),
            ("_nb_iter", "nb_iter", "nb_iter", 1, None, None),
            ("_adversarial_epochs", "adversarial_epochs", "adv_epochs", self._epochs // 2, int, None)
        ]

        self.autoset_params()

        if self._adversarial_epochs > self._epochs:
            raise Exception(f"The total epoch ({self._epochs}) "
                            f"is smaller than the adversarial epochs ({self._adversarial_epochs}).")

        if self._eps_iter is None:
            self._eps_iter = 2.5 * self._eps / self._nb_iter

        if self._batch_size < 1:
            self._batch_size = self._data.transactions

        self._ratings = self._data.train_dict

        self._sampler = cs.Sampler(self._data.i_train_dict)

        self._results_perturbation = {}

        self._model = MSAPMF_model(self._factors,
                                   self._learning_rate,
                                   self._l_w,
                                   self._l_b,
                                   self._eps,
                                   self._l_adv,
                                   self._eps_iter,
                                   self._nb_iter,
                                   self._num_users,
                                   self._num_items,
                                   self._seed)

    @property
    def name(self):
        return "MSAPMF" \
               + f"_{self.get_base_params_shortcut()}" \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        for it in self.iterate(self._epochs):
            user_adv_train = (self._epochs - it) <= self._adversarial_epochs
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                for batch in self._sampler.step(self._data.transactions, self._batch_size):
                    steps += 1
                    loss += self._model.train_step(batch, user_adv_train)
                    t.set_postfix({'(APR)-loss' if user_adv_train else '(BPR)-loss': f'{loss.numpy() / steps:.5f}'})
                    t.update()

            self.evaluate(it, loss.numpy() / (it + 1))

            if getattr(self._params.meta, "eval_perturbations", False):
                self.evaluate_perturbations(it)

    def evaluate_perturbations(self, it=None):
        if (it is None) or (not (it + 1) % self._validation_rate):

            for full_batch in self._sampler.step(self._data.transactions,
                                                 self._data.transactions):  # self._data.transactions
                self._model.build_msap_perturbation(full_batch, self._eps_iter, self._nb_iter)
                adversarial_iterative_recs = self.get_recommendations(self.evaluator.get_needed_recommendations(),
                                                                      adversarial=True)
                self._model.build_perturbation(full_batch)
                adversarial_single_recs = self.get_recommendations(self.evaluator.get_needed_recommendations(),
                                                                   adversarial=True)
            clean_result_dict = self._results[-1]
            adversarial_single_result_dict = self.evaluator.eval(adversarial_single_recs)
            adversarial_iterative_result_dict = self.evaluator.eval(adversarial_iterative_recs)

            self._results_perturbation[it] = {"clean": clean_result_dict,
                                              "adversarial_single": adversarial_single_result_dict,
                                              "adversarial_msap": adversarial_iterative_result_dict}

    def get_recommendations(self, k: int = 100, adversarial: bool = False):
        predictions_top_k_test = {}
        predictions_top_k_val = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.predict(offset, offset_stop, adversarial)
            recs_val, recs_test = self.process_protocol(k, predictions, offset, offset_stop)
            predictions_top_k_val.update(recs_val)
            predictions_top_k_test.update(recs_test)
        return predictions_top_k_val, predictions_top_k_test

    def get_results(self):
        if getattr(self._params.meta, "eval_perturbations", False):
            self.store_perturbation_results()

        return self._results[self.get_best_arg()]

    def store_perturbation_results(self):
        metrics = [m.name() for m in self.evaluator._metrics]
        attacked_single_metric = ['SSAP-' + m.name() for m in self.evaluator._metrics]
        attacked_iterative_metric = ['MSAP-' + m.name() for m in self.evaluator._metrics]
        df_adversarial_results = pd.DataFrame(columns=['Epoch', 'AdvEpoch',
                                                       'K'] + metrics + attacked_single_metric + attacked_iterative_metric)

        for it in self._results_perturbation.keys():
            for k in self._results_perturbation[it]['clean'].keys():
                df_adversarial_results.loc[len(df_adversarial_results)] = \
                    [it, self._adversarial_epochs, k] + \
                    list(self._results_perturbation[it]['clean'][k]['test_results'].values()) + \
                    list(self._results_perturbation[it]['adversarial_single'][k]['test_results'].values()) + \
                    list(self._results_perturbation[it]['adversarial_msap'][k]['test_results'].values())
        df_adversarial_results.to_csv(os.path.join(self._config.path_output_rec_performance,
                                                   f"adversarial-{self.name}.tsv"), index=False, sep='\t')
