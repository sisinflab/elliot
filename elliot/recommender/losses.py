import torch
import torch.nn.functional as F
from torch import nn, Tensor


class BPRLoss(nn.Module):
    """Bayesian Personalized Ranking pairwise loss: `-log(sigmoid(pos - neg))`,
    averaged over the batch and, when present, the negatives dimension.

    Supports both a single negative per positive (`pos_score` and `neg_score`
    share the same shape) and multiple negatives per positive (`neg_score`
    carries one extra trailing `neg_samples` dimension, against which
    `pos_score` broadcasts).
    """

    def forward(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        """
        Args:
            pos_score (Tensor): Positive item scores, shape `(*batch,)`.
            neg_score (Tensor): Negative item scores, shape `(*batch,)` for a
                single negative per positive, or `(*batch, neg_samples)` for
                multiple negatives per positive.

        Returns:
            Tensor: The scalar loss.
        """
        if neg_score.dim() > pos_score.dim():
            pos_score = pos_score.unsqueeze(-1)
        distance = pos_score - neg_score
        return F.softplus(-distance).mean()


class EmbLoss(nn.Module):
    """Batch-normalized L2 regularization loss over one or more embedding tensors."""

    def forward(self, *embeddings: Tensor) -> Tensor:
        """
        Args:
            *embeddings (Tensor): Embedding tensors, each shape `(batch, ..., dim)`,
                sharing the same batch size.

        Returns:
            Tensor: The scalar regularization loss.
        """
        reg = torch.zeros((), device=embeddings[0].device)
        for emb in embeddings:
            reg = reg + emb.pow(2).sum()
        return 0.5 * reg / embeddings[0].size(0)
