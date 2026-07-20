import random
import time
import numpy as np
import torch
from abc import ABC, abstractmethod
from tqdm import tqdm
from torch.utils.data import Dataset, TensorDataset

from elliot.utils import logging as elog
from elliot.utils.enums import SamplerType, SessionStrategy


class AbstractSampler(ABC):
    type: SamplerType

    def __init__(
        self,
        users,
        items,
        n_users,
        n_items,
        seed,
        logger=None,
        **kwargs
    ):
        self.logger = logger or elog.get_logger(self.__class__.__name__, seed=seed)

        self._users = users
        self._nusers = n_users
        self._items = items
        self._nitems = n_items

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

    def __init__(
        self,
        train_dict,
        transactions,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.events = transactions
        self._indexed_ratings = train_dict
        self._ui_dict = {u: list(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

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

    def __init__(
        self,
        train_dict,
        transactions,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.events = transactions
        self._indexed_ratings = train_dict
        self._ui_dict = {u: list(set(self._indexed_ratings[u])) for u in self._indexed_ratings}
        self._lui_dict = {u: len(v) for u, v in self._ui_dict.items()}

    def collate_fn(self, batch):
        self._r_shuffle(batch)

        tensors = tuple(
            torch.tensor(x, dtype=torch.long) for x in zip(*batch)
        )

        return tensors


class SessionSampler(AbstractSampler):
    """Base class for samplers operating on `Sessions`' flat item tape.

    Subclasses receive the same globally sorted (by user, session, timestamp)
    tape regardless of strategy: the only thing that changes between FLAT and
    SESSION_ONLY is which boundary array bounds a sequence (per-user vs.
    per-session), resolved once here so subclasses never branch on strategy.
    """

    type = SamplerType.SEQUENTIAL

    def __init__(
        self,
        flat_items,
        flat_users,
        flat_session,
        user_offsets,
        session_offsets,
        sparse,
        strategy=SessionStrategy.FLAT,
        max_seq_len=50,
        neg_samples=0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.flat_items = np.asarray(flat_items)
        self.flat_users = np.asarray(flat_users)
        self.flat_session = np.asarray(flat_session)
        self.user_offsets = np.asarray(user_offsets)
        self.session_offsets = np.asarray(session_offsets)
        self.sparse = sparse

        self.strategy = strategy if isinstance(strategy, SessionStrategy) else SessionStrategy(strategy)
        self.max_seq_len = max_seq_len
        self.neg_samples = neg_samples
        self.niid = self._nitems
        self.padding_token = self._nitems

        self._boundaries = (
            self.session_offsets if self.strategy == SessionStrategy.SESSION_ONLY else self.user_offsets
        )

        self._valid_target_indices = self._compute_valid_targets()
        self.events = len(self._valid_target_indices)

    def _compute_valid_targets(self):
        """A flat position is a valid next-item target iff it isn't the first
        position of its boundary segment (it needs at least one predecessor)."""
        n = len(self.flat_items)
        valid_mask = np.ones(n, dtype=bool)
        starts = self._boundaries[:-1]
        active_starts = starts[starts < n]
        valid_mask[active_starts] = False
        return np.arange(n)[valid_mask]

    def _boundary_id_of(self, idx):
        return int(self.flat_users[idx]) if self.strategy == SessionStrategy.FLAT else int(self.flat_session[idx])

    def _boundary_start_of(self, idx):
        return int(self._boundaries[self._boundary_id_of(idx)])

    def _build_padded_sequence(self, end_idx, boundary_start):
        start_idx = max(boundary_start, end_idx - self.max_seq_len)
        seq_array = self.flat_items[start_idx:end_idx]
        seq_len = len(seq_array)

        seq_tensor = torch.full((self.max_seq_len,), self.padding_token, dtype=torch.long)
        if seq_len:
            seq_tensor[:seq_len] = torch.from_numpy(seq_array.copy())

        return seq_tensor, seq_len

    def _sample_negatives(self, owner_user, k, exclude_item=None):
        u_start = self.sparse.indptr[owner_user]
        u_end = self.sparse.indptr[owner_user + 1]
        seen_items = self.sparse.indices[u_start:u_end]

        negatives = []
        while len(negatives) < k:
            cand = int(self._r_int(self.niid))
            if exclude_item is not None and cand == exclude_item:
                continue
            pos = np.searchsorted(seen_items, cand)
            if pos < len(seen_items) and seen_items[pos] == cand:
                continue
            negatives.append(cand)

        return negatives


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


class SequentialDataset(Dataset):
    """Lazy dataset for samplers whose `sample(it)` already returns a tuple of
    built tensors (e.g. padded sequences). Left to PyTorch's default collate,
    which stacks each tuple position independently.
    """

    def __init__(self, sampler):
        super().__init__()
        self.sampler = sampler

    def __len__(self):
        return self.sampler.events

    def __getitem__(self, idx):
        return self.sampler.sample(idx)


def build_dataset(sampler: AbstractSampler):
    match sampler.type:
        case SamplerType.TRADITIONAL:
            samples = sampler.sample_full()
            tensors = tuple(torch.tensor(x, dtype=torch.long) for x in zip(*samples))
            dataset = TensorDataset(*tensors)

        case SamplerType.PIPELINE:
            dataset = PipelineDataset(sampler)

        case SamplerType.SEQUENTIAL:
            dataset = SequentialDataset(sampler)

        case _:
            raise ValueError(f"Invalid sampler type {sampler.type}")

    collate_fn = getattr(sampler, 'collate_fn', None)
    if collate_fn is not None:
        setattr(dataset, 'collate_fn', collate_fn)

    return dataset
