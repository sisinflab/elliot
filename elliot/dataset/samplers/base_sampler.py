import random
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset

from elliot.utils import logging as elog
from elliot.utils.enums import SamplerType


class AbstractSampler(ABC):
    type: SamplerType

    def __init__(
        self,
        train_dict,
        transactions,
        users,
        items,
        n_users,
        n_items,
        seed,
        logger=None,
        **kwargs
    ):
        self.logger = logger or elog.get_logger(self.__class__.__name__, seed=seed)

        self.events = transactions

        self._users = users
        self._nusers = n_users
        self._items = items
        self._nitems = n_items

        self._indexed_ratings = train_dict
        self._ui_dict = {u: list(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

        np.random.seed(seed)
        random.seed(seed)

        self._r_int = np.random.randint
        self._r_choice = np.random.choice
        self._r_shuffle = random.shuffle
        self._r_sample = random.sample

    def sample_full(self):
        pass

    @abstractmethod
    def sample(self, it):
        raise NotImplementedError()

    def sample_eval(self, it):
        pass

    # def read_features(self, *args):
    #     return args
    #
    # def read_features_eval(self, *args):
    #     return args


class TraditionalSampler(AbstractSampler):
    type = SamplerType.TRADITIONAL

    def __init__(self, **params):
        super().__init__(**params)

    def sample_full(self, val=False):
        start = time.time()

        iter_data = tqdm(
            range(self.events),
            total=self.events,
            desc="Sampling",
            leave=False
        )
        samples = []
        sample_fn = self.sample if not val else self.sample_eval
        read_features_fn = self.read_features if not val else self.read_features_eval

        for it in iter_data:
            output = sample_fn(it)
            output = read_features_fn(*output)
            if isinstance(output, list):
                samples.extend(output)
            else:
                samples.append(output)

        self._r_shuffle(samples)

        end = time.time()

        self.logger.debug(
            "Completed sampling",
            extra={"context": {"duration_sec": round(end - start, 4), "events": len(samples)}}
        )

        return samples


class PipelineSampler(AbstractSampler):
    type = SamplerType.PIPELINE

    def __init__(self, **params):
        super().__init__(**params)

    def collate_fn(self, batch):
        self._r_shuffle(batch)

        tensors = tuple(
            torch.tensor(x, dtype=torch.long) for x in zip(*batch)
        )

        return tensors


class PipelineDataset(Dataset):
    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler
        self.m = getattr(sampler, 'm', 0)

    def __len__(self):
        return self.sampler.events * (self.m + 1)

    def __getitem__(self, idx):
        real_idx = idx // (self.m + 1)
        return self.sampler.sample(real_idx)


def build_dataset(sampler: AbstractSampler):
    match sampler.type:
        case SamplerType.TRADITIONAL:
            samples = sampler.sample_full()
            tensors = tuple(torch.tensor(x, dtype=torch.long) for x in zip(*samples))
            dataset = TensorDataset(*tensors)

        case SamplerType.PIPELINE:
            dataset = PipelineDataset(sampler)

        case _:
            raise ValueError(f"Invalid sampler type {sampler.type}")

    return dataset
