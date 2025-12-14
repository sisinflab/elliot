import inspect
import random
import time
import numpy as np
from abc import ABC, abstractmethod

from elliot.dataset import DataSet
from elliot.utils import logging as elog


class AbstractSampler(ABC):
    def __init__(
        self,
        train_dict,
        transactions,
        users,
        items,
        seed,
        logger=None,
        **kwargs
    ):
        np.random.seed(seed)
        random.seed(seed)
        self._r_int = np.random.randint
        self._r_choice = np.random.choice
        self._r_perm = np.random.permutation

        self.events = transactions

        self._users = users
        self._nusers = len(users)
        self._items = items
        self._nitems = len(items)

        self._indexed_ratings = train_dict
        self._ui_dict = {u: list(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}
        self.logger = logger or elog.get_logger(self.__class__.__name__, seed=seed)

    def initialize(self):
        start = time.time()
        samples = self._sample()
        end = time.time()
        self.logger.debug(
            "Completed sampling",
            extra={"context": {"duration_sec": round(end - start, 4), "events": getattr(self, "events", None)}}
        )
        return samples

    @abstractmethod
    def _sample(self, **kwargs):
        raise NotImplementedError()

    # def step_eval(self, *args):
    #     pass

    def _sample_eval(self, *args):
        pass


# class TraditionalSampler(AbstractSampler):
#     def __init__(self, seed, indexed_ratings=None):
#         super().__init__(seed, indexed_ratings)
#         self._cached_dataloaders = {}
#
#     def initialize(self):
#         cache_key = self.__class__.__name__
#         if cache_key in self._cached_dataloaders:
#             return self._cached_dataloaders[cache_key]
#
#         start = time.time()
#         samples = self._sample()
#         end = time.time()
#         print(f"Sampling has taken {end - start:.4f} seconds.")
#         arrays = tuple(map(np.array, samples))
#
#         dataloader = SimpleDataLoader(arrays, batch_size=self.batch_size, shuffle=True)
#         self._cached_dataloaders[cache_key] = dataloader
#
#         return dataloader

    # def prepare_output(self, *args):
    #     return args

    # def step(self):
    #     for batch_start in range(0, self.events, self.batch_size):
    #         batch_stop = min(batch_start + self.batch_size, self.events)
    #         current_batch_size = batch_stop - batch_start
    #         sample_out = self._sample(bs=batch_start, bsize=current_batch_size)
    #         # if isinstance(sample_out[0], (list, tuple, np.ndarray)):
    #         #     res = map(np.array, sample_out)
    #         # else:
    #         #     res = map(np.array, zip(*[self._sample(idx=i) for i in range(batch_start, batch_stop)]))
    #         yield tuple(self.prepare_output(*sample_out)) #tuple(r[:, None] for r in res)

    # @abstractmethod
    # def _sample(self, **kwargs):
    #     raise NotImplementedError()
    #

# class PipelineSampler(AbstractSampler):
#     def __init__(self, seed, indexed_ratings=None):
#         super().__init__(seed, indexed_ratings)
#         self._cached_dataloaders = {}

    # def _wrap_dataset(self, sample_fn):
    #     sampler = self
    #
    #     if inspect.isgeneratorfunction(sample_fn):
    #         class WrappedIterable(torch.utils.data.IterableDataset):
    #             def __iter__(self):
    #                 for res in sample_fn():
    #                     yield tuple(torch.tensor(f) for f in sampler.read_features(*res))
    #
    #         return WrappedIterable()
    #     else:
    #         class WrappedDataset(torch.utils.data.Dataset):
    #             def __len__(self):
    #                 return sampler.events
    #
    #             def __getitem__(self, idx):
    #                 res = tuple(torch.tensor(r) for r in sample_fn())
    #                 return tuple(torch.tensor(f) for f in sampler.read_features(*res))
    #
    #         return WrappedDataset()
    #
    # def initialize(self):
    #     cache_key = self.__class__.__name__
    #     if cache_key in self._cached_dataloaders:
    #         return self._cached_dataloaders[cache_key]
    #
    #     start = time.time()
    #     samples = self._sample()
    #     end = time.time()
    #     print(f"Sampling has taken {end - start:.4f} seconds.")
    #     tensors = tuple(torch.tensor(x) for x in samples)
    #
    #     dataset = TensorDataset(*tensors)
    #     dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    #     self._cached_dataloaders[cache_key] = dataloader
    #
    #     return dataloader

    # def step_eval(self):
    #     self.read_features = self.read_eval_features
    #     dataset = self._wrap_dataset(self._sample_eval)
    #     return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)
    #
    # def read_training_features(self, *args):
    #     return args
    #
    # def read_eval_features(self, *args):
    #     return args

    # @abstractmethod
    # def _sample(self, **kwargs):
    #     raise NotImplementedError()

# class SimpleDataLoader:
#     def __init__(self, arrays, batch_size, shuffle=True):
#         self.arrays = arrays
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.events = len(self.arrays[0])
#
#     def __iter__(self):
#         indices = list(range(self.events))
#         if self.shuffle:
#             np.random.shuffle(indices)
#
#         for batch_start in range(0, self.events, self.batch_size):
#             batch_stop = min(batch_start + self.batch_size, self.events)
#             batch_idx = indices[batch_start:batch_stop]
#             yield tuple(a[batch_idx] for a in self.arrays)
