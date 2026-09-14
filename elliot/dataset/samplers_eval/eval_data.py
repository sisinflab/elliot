from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm


class NegEvalDataset(Dataset):
    """Evaluation dataset pairing sampled negatives with the ground-truth positive(s)
    for each user, used when `NegativeSamplingConfig` is configured (as opposed to
    `FullEvalDataset`'s full-ranking evaluation).

    Args:
        num_users (int): Total number of users (or eval rows, for SESSION_ONLY
            evaluation) to evaluate.
        eval_neg_items (List[List[int]]): Negative item indices per row, already
            sampled. Under SESSION_ONLY evaluation this is the owning user's single,
            once-sampled negative list broadcast to every session row they own (see
            `DataSet._get_user_negatives()`), so that FLAT and SESSION_ONLY models are
            always scored against the same negatives.
        eval_pos_items (List[List[int]]): Ground-truth positive item indices per user.
        evaluation_set (str): Name of this fold's eval split ("test" or "validation").
            Defaults to "test".
        leave_one_out (bool): If True, only the last ground-truth positive per user is
            kept. Defaults to False.
    """

    def __init__(
        self,
        num_users: int,
        eval_neg_items: List[List[int]],
        eval_pos_items: List[List[int]],
        evaluation_set: str = "test",
        leave_one_out: bool = False
    ):
        # Initializing variables
        self.num_users = num_users
        self.leave_one_out = leave_one_out

        self._evaluation_set = evaluation_set

        self.eval_items = self._add_indices(eval_neg_items, eval_pos_items)

    def _add_indices(self, neg: List[List[int]], pos: List[List[int]]) -> Optional[List[torch.Tensor]]:
        """Add test or validation samples to the sampled negatives.

        Args:
            neg (List[List[int]]): Negative item indices per user.
            pos (List[List[int]]): Ground-truth positive item indices per user.

        Returns:
            Optional[List[torch.Tensor]]: Per-user tensors of negative items followed
                by the ground-truth positive(s), or None if `neg` is empty.
        """
        if not neg:
            return None

        final_items = []
        iter_data = tqdm(
            total=len(neg),
            desc=f"Adding {self._evaluation_set} items to sampled negatives",
            leave=False,
        )

        with iter_data as t:
            for neg_u, pos_u in zip(neg, pos):
                if not neg_u:
                    pos_u = []
                elif self.leave_one_out:
                    pos_u = [pos_u[-1]] if pos_u else []

                final_items.append(torch.tensor(neg_u + pos_u))
                t.update(1)

        return final_items

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, index: int) -> Tuple[int, torch.Tensor]:
        return index, self.eval_items[index]

    @staticmethod
    def collate_fn(batch: List[Tuple[int, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Collate a batch of (user index, items) pairs, padding the ragged item
        tensors to the batch's longest one.

        Args:
            batch (List[Tuple[int, torch.Tensor]]): The batch of (user index, items)
                pairs.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The batched user indices and the
                padded (`-1`-filled) item tensor.
        """
        user_indices, item_indices = zip(*batch)

        # User indices will be a list of ints, so we convert it
        user_indices = torch.tensor(list(user_indices))

        # We use the pad_sequence utility to pad item indices
        # in order to have all tensors of the same size
        item_indices = pad_sequence(
            item_indices,
            batch_first=True,
            padding_value=-1,
        )

        return user_indices, item_indices


class FullEvalDataset(Dataset):
    """Full-ranking evaluation dataset: one entry per user, with no sampled
    negatives (candidates are every item, ranked at evaluation time), used when no
    `NegativeSamplingConfig` is configured.

    Args:
        num_users (int): Total number of users (or eval rows, for SESSION_ONLY
            evaluation) to evaluate.
    """

    def __init__(self, num_users: int):
        # Initializing variables
        self.num_users = num_users

    def __len__(self) -> int:
        return self.num_users

    def __getitem__(self, index: int) -> int:
        return index

    @staticmethod
    def collate_fn(batch: List[int]) -> Tuple[torch.Tensor, None]:
        """Collate a batch of user indices; there is no per-user item tensor to pad.

        Args:
            batch (List[int]): The batch of user indices.

        Returns:
            Tuple[torch.Tensor, None]: The batched user indices, and `None` in place
                of an item tensor.
        """
        return torch.tensor(batch), None
