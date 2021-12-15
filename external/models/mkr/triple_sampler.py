"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Alberto Carlo Maria Mancino'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, alberto.mancino@poliba.it'

import random
import numpy as np


class Sampler:
    def __init__(self, item_entity, entity_to_idx, Xs, Xp, Xo, seed):
        random.seed(seed)
        mapped_entities = set(item_entity.values())
        self.triples = [(s, p, o) for s, p, o in zip(Xs, Xp, Xo) if s in mapped_entities]

        self.headDict = {}
        self.tailDict = {}
        for s, p, o in zip(Xs, Xp, Xo):
            self.headDict.setdefault((p, o), []).append(s)
            self.tailDict.setdefault((s, p), []).append(o)

        self.entity_total = list(range(len(entity_to_idx)))
        self.triples_idx = list(range(len(self.triples)))
        self.entity_item = {e: i for i, e in item_entity.items()}
        print('debug')

    def step(self, batch_size: int):
        s, p, o, i = [], [], [], []
        sn, pn, on, ii = [], [], [], []
        for _ in range(batch_size):
            idx = random.choice(self.triples_idx)
            idxn = random.choice(self.triples_idx)
            s_, p_, o_ = self.triples[idx]
            s.append(s_)
            p.append(p_)
            o.append(o_)
            i.append(self.entity_item[s_])

            sn_, pn_, on_ = self.triples[idxn]
            sn_, pn_, on_ = self.corrupt_tail_filter((sn_, pn_, on_))

            sn.append(sn_)
            pn.append(pn_)
            on.append(on_)
            ii.append(self.entity_item[sn_])

        return s, p, o, sn, pn, on

    def getTrainTripleBatch(self, triple_batch):
        negTripleList = [self.corrupt_head_filter(triple) if random.random() < 0.5
                         else self.corrupt_tail_filter(triple) for triple in
                         triple_batch]
        nh, nr, nt = zip(*negTripleList)
        return np.array(nh, dtype=np.int32), np.array(nr, dtype=np.int32), np.array(nt, dtype=np.int32)

    def corrupt_head_filter(self, triple):
        while True:
            newHead = random.choice(self.entity_total)
            if newHead == triple[0]:
                continue
            if self.headDict is not None:
                rt = (triple[1], triple[2])
                if newHead in self.headDict[rt]:
                    continue
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
                else:
                    break
            else:
                raise Exception("No tail dictionary found")
        return (triple[0], triple[1], newTail)
