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


class GBCELoss(nn.Module):
    """Generalised Binary Cross-Entropy (gBCE) loss, from gSASRec (Petrov and
    Macdonald, RecSys 2023): BCE over one positive and `neg_samples` uniformly
    sampled negatives, with the positive's sigmoid raised to the power `beta`
    to counter the overconfidence induced by negative sampling.

    `beta` is set through the calibration parameter `t` in [0, 1] as
    `beta = alpha * (t * (1 - 1 / alpha) + 1 / alpha) = 1 - t * (1 - alpha)`,
    `alpha = neg_samples / (num_items - 1)` being the negative sampling rate:
    `t = 0` (`beta = 1`) is the regular BCE loss, `t = 1` (`beta = alpha`) the
    fully calibrated one.

    Args:
        neg_samples (int): Number of negatives sampled per positive.
        num_items (int): Catalog size.
        t (float): Calibration parameter, in [0, 1].

    Raises:
        ValueError: If `neg_samples` is not positive or `t` is not in [0, 1].
    """

    def __init__(self, neg_samples: int, num_items: int, t: float):
        super().__init__()
        if neg_samples < 1:
            raise ValueError(f"gBCE needs at least one negative per positive, got {neg_samples}.")
        if not (0 <= t <= 1):
            raise ValueError(f"gBCE calibration parameter 't' must be in [0, 1], got {t}.")
        alpha = neg_samples / (num_items - 1)
        self.beta = 1 - t * (1 - alpha)

    def forward(self, pos_score: Tensor, neg_score: Tensor) -> Tensor:
        """
        Args:
            pos_score (Tensor): Positive item scores, shape `(*batch,)`.
            neg_score (Tensor): Negative item scores, shape `(*batch, neg_samples)`.

        Returns:
            Tensor: The scalar loss.
        """
        # -log(sigmoid(s)^beta) = beta * softplus(-s); -log(1 - sigmoid(s)) = softplus(s)
        pos_loss = self.beta * F.softplus(-pos_score)
        neg_loss = F.softplus(neg_score).sum(dim=-1)
        return ((pos_loss + neg_loss) / (neg_score.size(-1) + 1)).mean()


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
