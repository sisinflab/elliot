import torch
import torch.nn as nn
import typing as t
from collections import Counter
import math
import numpy as np
from enum import Enum

from elliot.recommender.base_trainer import Trainer, TraditionalTrainer, GeneralTrainer


class ModelType(Enum):
    BASE = 1
    TRADITIONAL = 2
    GENERAL = 3


def get_model(data, config, params, model_class):
    #model = model_class(data, params)
    if model_class.type == ModelType.BASE:
        trainer = Trainer
    elif model_class.type == ModelType.TRADITIONAL:
        trainer = TraditionalTrainer
    elif model_class.type == ModelType.GENERAL:
        trainer = GeneralTrainer
    return trainer(data, config, params, model_class)


class TFIDF:
    def __init__(self, map: t.Dict[int, t.List[int]]):
        self.__map = map
        self.__o = Counter(feature for feature_list in self.__map.values() for feature in feature_list )
        self.__maxi = max(self.__o.values())
        self.__total_documents = len(self.__map)
        self.__idfo = {k: math.log(self.__total_documents/v) for k, v in self.__o.items()}
        self.__tfidf = {}
        for k, v in self.__map.items():
            normalization = math.sqrt(sum([self.__idfo[i]**2 for i in v]))
            self.__tfidf[k] ={i:self.__idfo[i]/normalization for i in v}

    def tfidf(self):
        return self.__tfidf

    def get_profiles(self, ratings: t.Dict[int, t.Dict[int, float]]):
        profiles = {}
        profiles = {u: {f: profiles.get(u, {}).get(f, []) + [v] for i in items.keys() if i in self.__tfidf.keys() for f, v in self.__tfidf[i].items()} for u, items in ratings.items()}
        profiles = {u: {f: np.average(v) for f, v in f_dict.items()} for u, f_dict in profiles.items()}
        return profiles


class GaussianNoise(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        noise = torch.randn_like(x) * self.stddev
        return x + noise
