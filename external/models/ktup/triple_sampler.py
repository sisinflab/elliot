"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

import random

import numpy as np


class Sampler:
    def __init__(self, entity_to_idx, Xs, Xp, Xo, events
                 ):
        random.seed(42)
        self.events = events
        self.Xs, self.Xp, self.Xo = Xs, Xp, Xo
        # self.headDict = {(p, o): s for s, p, o in zip(Xs, Xp, Xo)}
        # self.tailDict = {(s, p): o for s, p, o in zip(Xs, Xp, Xo)}
        self.headDict = {}
        self.tailDict = {}
        for s, p, o in zip(Xs, Xp, Xo):
            self.headDict.setdefault((p, o), []).append(s)
            self.tailDict.setdefault((s, p), []).append(o)
        self.entity_total = list(range(len(entity_to_idx)))

    def step(self, batch_size: int):
        ntriples = len(self.Xs)
        # shuffled_list = random.sample(range(ntriples), self.events)
        shuffled_list = [random.choice(range(ntriples)) for _ in range(self.events)]

        for start_idx in range(0, self.events, batch_size):
            end_idx = min(start_idx + batch_size, self.events)
            ph, pr, pt = self.Xs[shuffled_list[start_idx:end_idx]], self.Xp[shuffled_list[start_idx:end_idx]], self.Xo[shuffled_list[start_idx:end_idx]]
            nh, nr, nt = self.getTrainTripleBatch(zip(ph, pr, pt))
            yield ph, pr, pt, nh, nr, nt

    def getTrainTripleBatch(self, triple_batch):
        negTripleList = [self.corrupt_head_filter(triple) if random.random() < 0.5
                         else self.corrupt_tail_filter(triple) for triple in
                         triple_batch]
        # yield u, pi, ni, each list contains batch size ids,
        # ph, pt, pr = getTripleElements(triple_batch)
        nh, nr, nt = zip(*negTripleList)
        return np.array(nh, dtype=np.int32), np.array(nr, dtype=np.int32), np.array(nt, dtype=np.int32)

    # Change the head of a triple randomly,
    # with checking whether it is a false negative sample.
    # If it is, regenerate.
    def corrupt_head_filter(self, triple):
        while True:
            newHead = random.choice(self.entity_total)
            if newHead == triple[0]:
                continue
            if self.headDict is not None:
                rt = (triple[1], triple[2])
                if newHead in self.headDict[rt]:
                    continue
                # for head_dict in headDicts:
                #     if tr in head_dict and newHead in head_dict[tr]:
                #         has_exist = True
                #         break
                # if has_exist:
                #     continue
                else:
                    break
            else:
                raise Exception("No head dictionary found")
        return (newHead, triple[1], triple[2])

    def corrupt_tail_filter(self, triple):
        while True:
            newTail = random.choice(self.entity_total)
            if newTail == triple[2]:
                continue
            if self.tailDict is not None:
                hr = (triple[0], triple[1])
                if newTail in self.tailDict[hr]:
                    continue
                # for head_dict in headDicts:
                #     if tr in head_dict and newHead in head_dict[tr]:
                #         has_exist = True
                #         break
                # if has_exist:
                #     continue
                else:
                    break
            else:
                raise Exception("No tail dictionary found")
        return (triple[0], triple[1], newTail)
