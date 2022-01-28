import os
import pickle
import sys
import typing as t
import numpy as np
from types import SimpleNamespace

from elliot.dataset.modular_loaders.abstract_loader import AbstractLoader


class TextualAttributeSequence(AbstractLoader):
    def __init__(self, users: t.Set, items: t.Set, ns: SimpleNamespace, logger: object):
        self.logger = logger
        self.textual_feature_file_path = getattr(ns, "textual_features", None)
        self.textual_feature_pretrain_path = getattr(ns, "pretrain_w2v", None)

        self.textual_features = self.check_items_in_file()

        self.users = users
        self.items = items

    def get_mapped(self) -> t.Tuple[t.Set[int], t.Set[int]]:
        return self.users, self.items

    def filter(self, users: t.Set[int], items: t.Set[int]):
        self.users = self.users & users
        self.items = self.items & items

    def create_namespace(self) -> SimpleNamespace:
        ns = SimpleNamespace()
        ns.__name__ = "TextualAttributeSequence"
        ns.object = self
        ns.textual_features = self.textual_features
        ns.load_word2vec_pretrain = self.load_word2vec_pretrain

        return ns

    def check_items_in_file(self):
        textual_features = None
        if self.textual_feature_file_path:
            textual_features = pickle.load(open(self.textual_feature_file_path, "rb"))
        return textual_features

    def load_word2vec_pretrain(self, emb_dim):
        vocab = self.textual_features['X_vocab']
        if self.textual_feature_pretrain_path:
            if os.path.isfile(self.textual_feature_pretrain_path):
                raw_word2vec = open(self.textual_feature_pretrain_path, 'r')
            else:
                print("Path (word2vec) is wrong!")
                sys.exit()

            word2vec_dic = {}
            all_line = raw_word2vec.read().splitlines()
            mean = np.zeros(emb_dim)
            count = 0
            for line in all_line:
                tmp = line.split()
                _word = tmp[0]
                _vec = np.array(tmp[1:], dtype=float)
                if _vec.shape[0] != emb_dim:
                    print("Mismatch the dimension of pre-trained word vector with word embedding dimension!")
                    sys.exit()
                word2vec_dic[_word] = _vec
                mean = mean + _vec
                count = count + 1

            mean = mean / count

            W = np.zeros((len(vocab) + 1, emb_dim))
            count = 0
            for _word, i in vocab:
                if _word in word2vec_dic:
                    W[i + 1] = word2vec_dic[_word]
                    count = count + 1
                else:
                    W[i + 1] = np.random.normal(mean, 0.1, size=emb_dim)

            print("%d words exist in the given pretrained model" % count)

            return W
        pass
