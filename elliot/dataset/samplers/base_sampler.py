import inspect
import random
import torch
import numpy as np
from abc import ABC, abstractmethod


class AbstractSampler(ABC):
    def __init__(self, seed, indexed_ratings):
        np.random.seed(seed)
        random.seed(seed)
        self._r_int = np.random.randint
        self.batch_size = 1
        self.events = 0
        if indexed_ratings is not None:
            self._indexed_ratings = indexed_ratings
            self._users = list(self._indexed_ratings.keys())
            self._nusers = len(self._users)
            self._items = list({k for a in self._indexed_ratings.values() for k in a.keys()})
            self._nitems = len(self._items)
            self._ui_dict = {u: list(set(indexed_ratings[u])) for u in indexed_ratings}
            self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def initialize(self):
        pass

    @abstractmethod
    def step(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def _sample(self, **kwargs):
        raise NotImplementedError()

    def step_eval(self, *args):
        pass

    def _sample_eval(self, *args):
        pass


class TraditionalSampler(AbstractSampler):
    def __init__(self, seed, indexed_ratings=None):
        super().__init__(seed, indexed_ratings)

    def prepare_output(self, *args):
        return args

    def step(self):
        for batch_start in range(0, self.events, self.batch_size):
            batch_stop = min(batch_start + self.batch_size, self.events)
            current_batch_size = batch_stop - batch_start
            sample_out = self._sample(bs=batch_start, bsize=current_batch_size)
            # if isinstance(sample_out[0], (list, tuple, np.ndarray)):
            #     res = map(np.array, sample_out)
            # else:
            #     res = map(np.array, zip(*[self._sample(idx=i) for i in range(batch_start, batch_stop)]))
            yield tuple(self.prepare_output(*sample_out)) #tuple(r[:, None] for r in res)

    @abstractmethod
    def _sample(self, **kwargs):
        raise NotImplementedError()


class PipelineSampler(AbstractSampler):
    def __init__(self, seed, indexed_ratings=None):
        super().__init__(seed, indexed_ratings)

    def _wrap_dataset(self, sample_fn):
        sampler = self

        if inspect.isgeneratorfunction(sample_fn):
            class WrappedIterable(torch.utils.data.IterableDataset):
                def __iter__(self):
                    for res in sample_fn():
                        yield tuple(torch.tensor(f) for f in sampler.read_features(*res))

            return WrappedIterable()
        else:
            class WrappedDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return sampler.events

                def __getitem__(self, idx):
                    res = tuple(torch.tensor(r) for r in sample_fn())
                    return tuple(torch.tensor(f) for f in sampler.read_features(*res))

            return WrappedDataset()

    def step(self):
        self.read_features = self.read_training_features
        dataset = self._wrap_dataset(self._sample)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def step_eval(self):
        self.read_features = self.read_eval_features
        dataset = self._wrap_dataset(self._sample_eval)
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

    def read_training_features(self, *args):
        return args

    def read_eval_features(self, *args):
        return args


class FakeSampler(AbstractSampler):
    def __init__(self):
        super().__init__(42, None)

    def step(self, *args):
        return True

    def _sample(self, **kwargs):
        return True
