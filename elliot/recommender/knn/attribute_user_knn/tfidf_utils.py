import typing as t
from collections import Counter
import math

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
        profiles = {u: {f: profiles.get(u, {}).get(f, 0) + v for i in items.keys() if i in self.__tfidf.keys() for f, v in self.__tfidf[i].items()} for u, items in ratings.items()}
        profiles = {u: {f: v/len(ratings[u]) for f, v in f_dict.items()} for u, f_dict in profiles.items()}
        # print(profiles[0])
        # for u, items in ratings.items():
        #     profiles[u] = {}
        #     profiles[u] = {f: profiles[u].get(f, 0) for i in items.keys() for f in self.__tfidf[i]}
        return profiles
