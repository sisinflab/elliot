import typing as t
from collections import Counter
import math


class TFIDF:
    def __init__(self, map: t.Dict[int, t.List[int]]):
        self.__map = map
        self.__o = Counter(feature for feature_list in self.__map.values() for feature in feature_list)
        self.__maxi = max(self.__o.values())
        self.__total_documents = len(self.__map)
        self.__idfo = {k: math.log(self.__total_documents / v) for k, v in self.__o.items()}
        self.__tfidf = {}
        for k, v in self.__map.items():
            normalization = math.sqrt(sum([self.__idfo[i] ** 2 for i in v]))
            self.__tfidf[k] = {i: self.__idfo[i] / normalization for i in v}

    def tfidf(self):
        return self.__tfidf
