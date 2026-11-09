from typing import List, Optional, Tuple
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from elliot.evaluation.protocol.targets import EvalTargets


class NegEvalDataset(Dataset):
    """Evaluation dataset pairing sampled negatives with the ground-truth positive(s)
    for each eval row, used when `evaluation.negative_sampling` is configured (as
    opposed to `FullEvalDataset`'s full-ranking evaluation).

    Args:
        rows (EvalTargets): The eval rows to evaluate; each row's target item(s) are
            all candidates, exactly as all of them are relevant for the evaluator.
        eval_neg_items (List[List[int]]): Negative item indices per row, already
            sampled: the owning user's single, once-sampled negative list broadcast
            to every row they own (see `EvalProtocol.sample_user_negatives()`), so that
            every session strategy and target is scored against the same negatives.
        evaluation_set (str): Name of this fold's eval split ("test" or "validation").
            Defaults to "test".
    """

    def __init__(
        self,
        rows: EvalTargets,
        eval_neg_items: List[List[int]],
        evaluation_set: str = "test"
    ):
        # Initializing variables
        self.rows = rows

        self._evaluation_set = evaluation_set

        self.eval_items = self._add_indices(eval_neg_items, rows.target_items)

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

                final_items.append(torch.tensor(neg_u + pos_u))
                t.update(1)

        return final_items

    def __len__(self) -> int:
        return self.rows.n_rows

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
    """Full-ranking evaluation dataset: one entry per eval row, with no sampled
    negatives (candidates are every item, ranked at evaluation time), used when no
    `evaluation.negative_sampling` is configured.

    Args:
        rows (EvalTargets): The eval rows to evaluate.
    """

    def __init__(self, rows: EvalTargets):
        # Initializing variables
        self.rows = rows

    def __len__(self) -> int:
        return self.rows.n_rows

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
