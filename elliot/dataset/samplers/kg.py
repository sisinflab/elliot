from typing import Any, Dict, Iterable, Set, Tuple
import numpy as np

from elliot.dataset.samplers.base_sampler import PipelineSideInfoSampler
from elliot.utils.registry import sampler_registry


@sampler_registry.register()
class KGTriplesSampler(PipelineSideInfoSampler):
    """Samples `(head, relation, positive_tail, negative_tail)` quadruples directly
    from a knowledge graph's `(head, relation, tail)` triples, one per event, with
    the negative tail uniformly sampled among entities that are never an observed
    tail for that `(head, relation)` pair.

    Fully decoupled from interaction data: it draws only from the KG itself and
    ignores the `train_dict`/`users`/`items` a `sampler_registry.get()` call always
    forwards (see `PipelineSideInfoSampler`).

    Args:
        kg_heads (Iterable[int]): Head entity id per KG triple.
        kg_relations (Iterable[int]): Relation id per KG triple.
        kg_tails (Iterable[int]): Tail entity id per KG triple.
        n_entities (int): Total number of KG entities, for negative-tail sampling.
        **kwargs (Any): Forwarded to `PipelineSideInfoSampler.__init__`.
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

        self._kg_heads = np.asarray(kg_heads)
        self._kg_relations = np.asarray(kg_relations)
        self._kg_tails = np.asarray(kg_tails)
        self._n_entities = n_entities

        self.events = len(self._kg_heads)

        # (head, relation) -> observed tails, excluded when sampling a negative tail
        self._hr_tails: Dict[Tuple[int, int], Set[int]] = {}
        for h, r, t in zip(self._kg_heads, self._kg_relations, self._kg_tails):
            self._hr_tails.setdefault((int(h), int(r)), set()).add(int(t))

    def sample(self, it: int) -> Tuple[int, int, int, int]:
        """Build the `(head, relation, positive_tail, negative_tail)` quadruple for
        KG triple `it`.

        Args:
            it (int): KG triple index.

        Returns:
            Tuple[int, int, int, int]: The `(head, relation, positive_tail,
                negative_tail)` quadruple.
        """
        h = int(self._kg_heads[it])
        r = int(self._kg_relations[it])
        pos_t = int(self._kg_tails[it])

        pos_tails = self._hr_tails[(h, r)]
        neg_t = self._r_int(self._n_entities)
        while neg_t in pos_tails:
            neg_t = self._r_int(self._n_entities)

        return h, r, pos_t, neg_t
