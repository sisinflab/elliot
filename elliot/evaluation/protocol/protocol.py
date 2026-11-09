from typing import Dict, List, Optional, Set
import logging as pylog
from torch.utils.data import Dataset, DataLoader

from elliot.dataset import DataSet
from elliot.evaluation.protocol.targets import EvalTargets
from elliot.evaluation.protocol.candidates import NegativeSampler
from elliot.evaluation.protocol.datasets import NegEvalDataset, FullEvalDataset
from elliot.namespace import ExperimentConfig
from elliot.utils import logging
from elliot.utils.enums import EvalTarget, SessionStrategy


class EvalProtocol:
    """Evaluation protocol of one fold, shared by every model evaluated on it. Lazily
    builds (and caches) the `EvalTargets` rows of each requested session strategy and
    their evaluation `Dataset`, full-ranking or negative-sampled via `NegativeSampler`
    (restricted to the users of `evaluation.protocol.user_filter`, if any).

    It lives as long as its fold's `DataSet` does, not as long as a single trainer:
    rows are built once per session strategy, and negatives once per fold, so that
    models evaluated under different session strategies (FLAT and SESSION_ONLY) on
    the same fold are always scored against the exact same negative items.

    Args:
        config (ExperimentConfig): Experiment configuration object.
        dataset (DataSet): The fold's train/eval data.
    """

    def __init__(self, config: ExperimentConfig, dataset: DataSet):
        # Initializing variables
        self.config = config
        self.eval_protocol_config = config.evaluation.protocol
        self.dataset = dataset

        self._cached_datasets: Dict[SessionStrategy, Dataset] = {}

        # Negatives sampled once per train user and cached. Built lazily, on the
        # first call to `sample_user_negatives()`
        self._user_negatives: Optional[List[List[int]]] = None

    @property
    def target(self) -> EvalTarget:
        """The evaluation target of this fold's eval rows: the configured one on the
        test set, while validation keeps a single, deterministic and cheap target per
        sequence (LAST), unless every item is a target (ALL).
        """
        target = self.eval_protocol_config.target

        if self.dataset.evaluation_set != "test" and target != EvalTarget.ALL:
            target = EvalTarget.LAST

        return target

    @property
    def mask_seen(self) -> bool:
        """Whether already-seen items are excluded from full-ranking recommendations."""
        return self.eval_protocol_config.mask_seen

    def get_eval_dataloader(
        self,
        batch_size: int = 1024,
        session_strategy: Optional[SessionStrategy] = None
    ) -> DataLoader:
        """Build the evaluation dataloader for this fold, over the (possibly cached)
        evaluation dataset of the requested session strategy, honoring the configured
        negative-sampling strategy and evaluation target. Its dataset holds the
        `EvalTargets` rows it iterates over, as `dataloader.dataset.rows`.

        Args:
            batch_size (int): Batch size for the returned dataloader. Defaults to 1024.
            session_strategy (SessionStrategy, optional): SESSION_ONLY to request
                per-eval-session evaluation, which needs a dataset loaded with
                SESSION_ONLY segmentation. Defaults to None (FLAT).

        Returns:
            DataLoader: The evaluation dataloader.

        Raises:
            ValueError: If SESSION_ONLY evaluation is requested on a dataset that
                wasn't segmented into sessions.
        """
        strategy = SessionStrategy(session_strategy or SessionStrategy.FLAT)

        if (
            strategy == SessionStrategy.SESSION_ONLY
            and self.config.data_config.session_strategy != SessionStrategy.SESSION_ONLY
        ):
            raise ValueError("SESSION_ONLY evaluation needs a dataset segmented into sessions: "
                             "set `data_config.session_strategy: session_only`.")

        # One cached dataset per session strategy: the target is the same for all
        if strategy not in self._cached_datasets:
            self._cached_datasets[strategy] = self._build_eval_dataset(strategy, self.target)

        dataset = self._cached_datasets[strategy]

        return DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=dataset.collate_fn,
            shuffle=False,
        )

    def _build_eval_dataset(self, strategy: SessionStrategy, target: EvalTarget) -> Dataset:
        """Build the `EvalTargets` rows for the given session strategy and target, and
        their evaluation `Dataset`: full-ranking or negative-sampled (every row
        drawing from its owner's entry in the per-user negatives cache).

        Args:
            strategy (SessionStrategy): FLAT or SESSION_ONLY.
            target (EvalTarget): The evaluation target.

        Returns:
            Dataset: The evaluation dataset over the eval rows (`dataset.rows`), ready
                to be wrapped in a `DataLoader`.
        """
        rows = EvalTargets(
            eval_data=self.dataset.eval_set.dataframe,
            history=self.dataset.train_set.dataframe,
            seen=self.dataset.train_set.sparse_counts,
            mappings=self.dataset.get_mappings(),
            inv_mappings=self.dataset.get_inverse_mappings(),
            strategy=strategy,
            target=target,
            seed=self.config.random_seed,
            align_with_sessions=self.config.mixes_session_strategies,
            user_filter=self.eval_protocol_config.user_filter
        )

        # Negative-sampled evaluation: rows don't own a private negative sample,
        # every row broadcasts its owning user's single, once-sampled list
        if self.eval_protocol_config.negative_sampling is not None:
            if self._user_negatives is None:
                self.sample_user_negatives()

            eval_dataset = NegEvalDataset(
                rows=rows,
                eval_neg_items=[self._user_negatives[u] for u in rows.owner_users],
                evaluation_set=self.dataset.evaluation_set,
            )

        # Full-ranking evaluation: no negatives, every item is a candidate
        else:
            eval_dataset = FullEvalDataset(rows=rows)

        return eval_dataset

    def sample_user_negatives(self):
        """Sample (and cache) this fold's negative items once, over the whole
        train-user domain, regardless of which session strategy is requested first.
        Models evaluated with FLAT and SESSION_ONLY on this same fold must be scored
        against identical negatives per user for their metrics to be comparable, so
        `_build_eval_dataset()` never re-samples: it always reads from (and broadcasts)
        this single cache.
        """
        if self.eval_protocol_config.negative_sampling is not None:
            sampler = NegativeSampler(
                neg_sampling_config=self.eval_protocol_config.negative_sampling,
                mappings=self.dataset.get_mappings(),
                inv_mappings=self.dataset.get_inverse_mappings(),
                num_users=self.dataset.train_set.dims[0],
                num_items=self.dataset.train_set.dims[1],
                train_pos_items=self.dataset.train_set.get_positive_items(),
                eval_pos_items=self.dataset.eval_set.get_positive_items(),
                evaluation_set=self.dataset.evaluation_set,
                fold_index=self.dataset.fold_index,
            )
            self._user_negatives = sampler.sample()


def prepare_protocols(
    config: ExperimentConfig,
    val_data: List[List[DataSet]],
    main_data: List[DataSet]
):
    """Attach one `EvalProtocol` to every fold's `DataSet` (as `dataset.protocol`), and
    sample each fold's negatives eagerly, so that no evaluation call pays for it.
    A `DataSet` listed more than once (a test fold standing in for its missing
    validation folds) is prepared once, its single protocol serving both places.

    Args:
        config (ExperimentConfig): Experiment configuration object.
        val_data (List[List[DataSet]]): Per-test-fold lists of validation-fold
            `DataSet`s, as returned by `DataSetLoader.build()`.
        main_data (List[DataSet]): Per-test-fold `DataSet`s, as returned by
            `DataSetLoader.build()`.
    """
    logger = logging.get_logger(
        "EvalProtocol", pylog.CRITICAL if config.config_test else pylog.DEBUG
    )
    logger.info("Preparing evaluation protocols")

    prepared: Set[int] = set()

    def prepare(dataset: DataSet):
        if id(dataset) not in prepared:
            dataset.protocol = EvalProtocol(config, dataset)
            dataset.protocol.sample_user_negatives()
            prepared.add(id(dataset))

    for p1, (folds, main) in enumerate(zip(val_data, main_data)):
        # Test level
        logger.info(f"Test Fold {p1}")
        prepare(main)

        for p2, fold in enumerate(folds):
            # Validation level
            logger.info(f"Test Fold {p1} - Validation Fold {p2}")
            prepare(fold)
