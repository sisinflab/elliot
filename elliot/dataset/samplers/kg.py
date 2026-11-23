from typing import Any, Iterable, Tuple
import numpy as np
import torch

from elliot.dataset.samplers.base_sampler import KeySet, SideInfoSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class KGTriplesSampler(SideInfoSampler):
    """Samples `(head, relation, positive_tail, negative_tail)` quadruples directly
    from a knowledge graph's `(head, relation, tail)` triples, one per event, with
    the negative tail uniformly sampled among entities that are never an observed
    tail for that `(head, relation)` pair.

    Fully decoupled from interaction data: it draws only from the KG itself and
    ignores the `train_dict`/`users`/`items` a `sampler_registry.get()` call always
    forwards (see `SideInfoSampler`).

    Args:
        kg_heads (Iterable[int]): Head entity id per KG triple.
        kg_relations (Iterable[int]): Relation id per KG triple.
        kg_tails (Iterable[int]): Tail entity id per KG triple.
        n_entities (int): Total number of KG entities, for negative-tail sampling.
        **kwargs (Any): Forwarded to `SideInfoSampler.__init__`.
    """

    def __init__(
        self,
        kg_heads: Iterable[int],
        kg_relations: Iterable[int],
        kg_tails: Iterable[int],
        n_entities: int,
        **kwargs: Any
    ):
        super().__init__(**kwargs)

        heads = np.asarray(kg_heads, dtype=np.int64)
        relations = np.asarray(kg_relations, dtype=np.int64)
        tails = np.asarray(kg_tails, dtype=np.int64)
        self._n_entities = n_entities

        self.events = len(heads)

        # One group per distinct (head, relation) pair, whose observed tails are
        # excluded when sampling a negative tail
        hr_pairs = np.stack((heads, relations), axis=1)
        unique_pairs, hr_group = np.unique(hr_pairs, axis=0, return_inverse=True)
        hr_group = hr_group.reshape(-1).astype(np.int64)

        self._register_state("_kg_heads", heads)
        self._register_state("_kg_relations", relations)
        self._register_state("_kg_tails", tails)
        self._register_state("_kg_hr_group", hr_group)
        self._register_state("_hr_tails", KeySet.from_pairs(hr_group, tails, len(unique_pairs), n_entities))

    def sample(self, indices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the `(head, relation, positive_tail, negative_tail)` quadruples for
        KG triples `indices`.

        Args:
            indices (torch.Tensor): KG triple indices.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Heads,
                relations, positive tails and negative tails, int64, shape `(batch,)` each.
        """
        neg_tails = self._draw_negatives(
            self._kg_hr_group[indices], 1, self._n_entities, seen=self._hr_tails
        )[:, 0]

        return self._kg_heads[indices], self._kg_relations[indices], self._kg_tails[indices], neg_tails
