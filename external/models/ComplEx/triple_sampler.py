
import numpy as np
import random
from typing import Tuple
from types import SimpleNamespace


class TripleSampler:
    def __init__(self,
                 side: SimpleNamespace,
                 random_seed: int = 42) -> None:
        self.random_state = random.Random(random_seed)
        self.Xs = side.Xs
        self.Xp = side.Xp
        self.Xo = side.Xo

        self.Xi = np.arange(start=0, stop=self.Xs.shape[0], dtype=np.int32)

        assert np.allclose(self.Xs.shape, self.Xp.shape)
        assert np.allclose(self.Xs.shape, self.Xo.shape)

        self.nb_examples = self.Xs.shape[0]

    def step(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        curriculum_order = self.random_state.sample(range(self.nb_examples), self.nb_examples)

        for start_idx in range(0, self.nb_examples, batch_size):
            end_idx = min(start_idx + batch_size, self.nb_examples)
            yield self.Xp[curriculum_order[start_idx: end_idx]], \
                  self.Xs[curriculum_order[start_idx: end_idx]], \
                  self.Xo[curriculum_order[start_idx: end_idx]], \
                  self.Xi[curriculum_order[start_idx: end_idx]],
