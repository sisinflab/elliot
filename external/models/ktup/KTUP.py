"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from tqdm import tqdm
import numpy as np
import typing as t

from elliot.dataset.samplers import custom_sampler as cs
from elliot.evaluation.evaluator import Evaluator
from elliot.recommender import BaseRecommenderModel
from .KTUPModel import jtup
from elliot.recommender.base_recommender_model import init_charger
from elliot.utils.write import store_recommendation
from elliot.recommender.knowledge_aware.kaHFM_batch.tfidf_utils import TFIDF
from elliot.recommender.recommender_utils_mixin import RecMixin

np.random.seed(42)


class KTUP(RecMixin, BaseRecommenderModel):
    @init_charger
    def __init__(self, data, config, params, *args, **kwargs):
        """
        Create a BPR-MF instance.
        (see https://arxiv.org/pdf/1205.2618 for details about the algorithm design choices).

        Args:
            data: data loader object
            params: model parameters {embed_k: embedding size,
                                      [l_w, l_b]: regularization,
                                      lr: learning rate}
        """
        self._random = np.random

        self._ratings = self._data.train_dict
        self._sampler = cs.Sampler(self._data.i_train_dict)

        # autoset params
        self._params_list = [
            ("_l2_lambda", "l2_lambda", "l2", 0, None, None),
            ("_embedding_size", "embedding_size", "es", 100, int, None),
            ("_learning_rate", "lr", "lr", 0.001, None, None),
            ("_joint_ratio", "joint_ratio", "jr", 0.7, None, None),
            ("_L1", "L1_flag", "l1", True, None, None),
            ("_norm_lambda", "norm_lambda", "nl", 1, None, None),
            ("_kg_lambda", "kg_lambda", "kgl", 1, None, None),
            ("_use_st_gumbel", "use_st_gumbel", "gum", False, None, None),
        ]
        self.autoset_params()

        feature_names_path = self._config.data_config.side_information.features
        feature_names: t.Dict[int, t.List[str, str]] = self.load_feature_names(feature_names_path)

        self._relations = list({r for r, _ in feature_names.values()})
        self._private_relations = {p: u for p, u in enumerate(self._relations)}
        self._public_relations = {v: k for k, v in self._private_relations.items()}

        self._entities = list({o for _, o in feature_names.values()})
        self._private_entities = {p: u for p, u in enumerate(self._entities)}
        self._public_entities = {v: k for k, v in self._private_entities.items()}

        self._step_to_switch = self._joint_ratio * 10

        mapping = self.load_mapping("/media/cheggynho/WalterBackup03/KARS_pomo/kvae/code/data/categorical_dbpedia_ml1m/mapping.tsv")


        item2entity = {}
        entity2item = {}
        for p, item in enumerate(set(mapping.keys()) - set(self._entities), len(set(self._entities))):
            self._private_entities[p] = item
            self._public_entities[item] = p
            item2entity[mapping[item]] = p
            entity2item[p] = mapping[item]

        self._tfidf_obj = TFIDF(self._data.side_information_data.feature_map)
        self._tfidf = self._tfidf_obj.tfidf()
        self._user_profiles = self._tfidf_obj.get_profiles(self._ratings)

        self._user_factors = \
            np.zeros(shape=(len(self._data.users), len(self._data.features)))
        self._item_factors = \
            np.zeros(shape=(len(self._data.items), len(self._data.features)))

        for i, f_dict in self._tfidf.items():
            if i in self._data.items:
                for f, v in f_dict.items():
                    self._item_factors[self._data.public_items[i]][self._data.public_features[f]] = v

        for u, f_dict in self._user_profiles.items():
            for f, v in f_dict.items():
                self._user_factors[self._data.public_users[u]][self._data.public_features[f]] = v

        self._iteration = 0
        # self.evaluator = Evaluator(self._data, self._params)
        if self._batch_size < 1:
            self._batch_size = self._num_users

        ######################################

        self._dropout_rate = 1. - self._dropout_rate
        #
        self._model = jtup(self._L1, self._embedding_size, self._num_users, self._num_items, self._entity_total,
                           self._relation_total, self._i_map, self._kg_map)



    @property
    def name(self):
        return "kTUP" \
               + "_e:" + str(self._epochs) \
               + "_bs:" + str(self._batch_size) \
               + f"_{self.get_params_shortcut()}"

    def train(self):
        if self._restore:
            return self.restore_weights()

        best_metric_value = 0
        self._update_count = 0
        for it in range(self._epochs):
            loss = 0
            steps = 0
            with tqdm(total=int(self._data.transactions // self._batch_size), disable=not self._verbose) as t:
                if it % 10 < self._step_to_switch:
                    for batch in self._sampler.step(self._data.transactions, self._batch_size):
                        steps += 1
                        loss += self._model.train_step_rec(batch, is_rec=True)
                else:
                    for batch in self._sampler.step(self._data.transactions, self._batch_size):
                        steps += 1
                        loss += self._model.train_step_kg(batch, is_rec=False, kg_lambda=self.__kg_lambda)
                t.set_postfix({'loss': f'{loss.numpy() / steps:.5f}'})
                t.update()

            if not (it + 1) % self._validation_rate:
                recs = self.get_recommendations(self.evaluator.get_needed_recommendations())
                result_dict = self.evaluator.eval(recs)
                self._results.append(result_dict)

                print(f'Epoch {(it + 1)}/{self._epochs} loss {loss:.3f}')

                if self._results[-1][self._validation_k]["val_results"][self._validation_metric] > best_metric_value:
                    print("******************************************")
                    best_metric_value = self._results[-1][self._validation_k]["val_results"][self._validation_metric]
                    if self._save_weights:
                        self._model.save_weights(self._saving_filepath)
                    if self._save_recs:
                        store_recommendation(recs, self._config.path_output_rec_result + f"{self.name}-it:{it + 1}.tsv")

    def get_recommendations(self, k: int = 100):
        predictions_top_k = {}
        for index, offset in enumerate(range(0, self._num_users, self._batch_size)):
            offset_stop = min(offset + self._batch_size, self._num_users)
            predictions = self._model.get_recs(
                (
                    np.repeat(np.array(list(range(offset, offset_stop)))[:, None], repeats=self._num_items, axis=1),
                    np.array([self._i_items_set for _ in range(offset, offset_stop)])
                 )
            )
            v, i = self._model.get_top_k(predictions, self.get_train_mask(offset, offset_stop), k=k)
            items_ratings_pair = [list(zip(map(self._data.private_items.get, u_list[0]), u_list[1]))
                                  for u_list in list(zip(i.numpy(), v.numpy()))]
            predictions_top_k.update(dict(zip(map(self._data.private_users.get,
                                                  range(offset, offset_stop)), items_ratings_pair)))
        return predictions_top_k

    def load_feature_names(self, infile, separator='\t'):
        feature_names = {}
        with open(infile, "r") as file:
            for line in file:
                line = line.split(separator)
                pattern = line[1].split('><')
                pattern[0] = pattern[0][1:]
                pattern[1] = pattern[1][:-2]
                feature_names[int(line[0])] = pattern
        return feature_names

    def load_mapping(self, infile, separator="\t"):
        mapping = {}
        with open(infile, "r") as fin:
            for line in fin:
                line = line.rstrip("\n").split(separator)
                mapping[line[1]] = int(line[0])
        return mapping