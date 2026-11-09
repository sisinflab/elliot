from typing import Any, Callable, Dict, List, Optional, Literal
from pydantic import Field, model_validator

from elliot.namespace.common import BaseConfig
from elliot.namespace.read_config import TabularReaderConfig
from elliot.namespace.write_config import TabularWriterConfig
from elliot.utils import import_submodules
from elliot.utils.enums import EvalTarget, NegativeSamplingStrategy
from elliot.utils.folder import path_joiner
from elliot.utils.registry import metric_registry


class NegativeSamplingConfig(BaseConfig):
    """Negative sampling configuration.

    Attributes:
        strategy (NegativeSamplingStrategy): Negative sampling strategy to use.
        num_negatives (int): Number of negative samples. Defaults to 99, min is 1.
        save_on_disk (bool): Whether to save sampling results to disk. Defaults to False.
        save_folder (str, optional): Path to the folder where negative samples files will be saved.
        read_folder (str, optional): Path to the folder containing negative samples files;
            required for `fixed` strategy.
        reader (TabularReaderConfig): Reading configuration.
        writer (TabularWriterConfig): Writing configuration.
    """

    strategy: NegativeSamplingStrategy
    num_negatives: int = Field(default=99, ge=1)
    save_on_disk: bool = False
    save_folder: Optional[str] = path_joiner("..", "data", "{0}")
    read_folder: Optional[str] = None
    reader: TabularReaderConfig = Field(default_factory=TabularReaderConfig, exclude=True)
    writer: TabularWriterConfig = Field(default_factory=TabularWriterConfig, exclude=True)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "NegativeSamplingConfig":
        """Ensure required fields are set for the selected negative sampling strategy.

        Returns:
            NegativeSamplingConfig: The object itself.
        """
        if self.strategy == NegativeSamplingStrategy.FIXED and self.read_folder is None:
            raise AttributeError(f"Attribute `read_folder` must be provided "
                                 f"with '{self.strategy.value}' strategy.")

        return self


class UserFilterConfig(BaseConfig):
    """User filter configuration.

    Attributes:
        path (str, optional): Path to a file specifying users to include in evaluation.
        id_space (Literal["public", "private"]): Identifier space for the user filter file.
            Defaults to "public".
        reader (TabularReaderConfig): Reader configuration.
    """

    path: Optional[str] = None
    id_space: Literal["public", "private"] = "public"
    reader: TabularReaderConfig = Field(default_factory=TabularReaderConfig, exclude=True)


class ProtocolConfig(BaseConfig):
    """Protocol configuration.

    Attributes:
        target (EvalTarget, optional): Which eval item(s) each evaluation row predicts, 
            given everything known before them. Defaults to None.
        mask_seen (bool): Whether to exclude already seen items from full-ranking recommendations.
            Defaults to True.
        negative_sampling (NegativeSamplingConfig, optional): Negative sampling configuration.
            Defaults to None.
        user_filter (UserFilterConfig): User filter configuration. Defaults to no filter
            (every user is evaluated).
    """

    target: Optional[EvalTarget] = None
    mask_seen: bool = True
    negative_sampling: Optional[NegativeSamplingConfig] = None
    user_filter: UserFilterConfig = Field(default_factory=UserFilterConfig)


class EvaluationConfig(BaseConfig):
    """Evaluation configuration.

    Attributes:
        protocol (ProtocolConfig): Protocol-level configuration.
        cutoffs (List[int]): List of cutoff values used for evaluation metrics. Defaults to [].
        simple_metrics (List[str]): List of simple evaluation metric names. Defaults to [].
        complex_metrics (List[Dict[str, dict]]): List of complex evaluation metric configurations. Defaults to [].
        relevance_threshold (int): Minimum relevance value to consider an interaction relevant. Defaults to 0.
        paired_ttest (Dict[str, dict]): Configuration for paired t-test comparisons. Defaults to {}.
        wilcoxon_test (Dict[str, dict]): Configuration for Wilcoxon signed-rank tests. Defaults to {}.
        accelerate (bool, optional): Whether to use accelerate for mixed precision training. Defaults to None.
        accelerate_verify (bool): Whether to verify the model after training. Defaults to True.
        accelerate_verify_once (bool): Whether to verify the model only once. Defaults to True.
        accelerate_tolerance (float): Tolerance for the verification check. Defaults to 1e-6.
        accelerate_device (str, optional): Device to use for accelerate. Defaults to None.
    """
    
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    cutoffs: List[int] = []
    simple_metrics: List[str] = []
    complex_metrics: List[Dict[str, dict]] = []
    relevance_threshold: int = 0
    paired_ttest: Dict[str, dict] = {}
    wilcoxon_test: Dict[str, dict] = {}
    accelerate: Optional[bool] = None
    accelerate_verify: bool = True
    accelerate_verify_once: bool = True
    accelerate_tolerance: float = 1e-6
    accelerate_device: Optional[str] = None

    @model_validator(mode="after")
    def validate_metrics(self) -> "EvaluationConfig":
        """Validate metrics.

        Returns:
            EvaluationConfig: The object itself with validated metrics.
        """
        import_submodules("elliot.evaluation.metrics")

        self.simple_metrics = self._validate_metric_list(self.simple_metrics)
        self.complex_metrics = self._validate_metric_list(self.complex_metrics, lambda m: m["name"])

        return self

    def _validate_metric_list(self, metrics: List[Any], name_fn: Optional[Callable] = None) -> List[Any]:
        """Filter and validate a list of metrics against a registry.

        Args:
            metrics (List[Any]): List of metric names to verify and filter.
            name_fn (Callable, optional): Function to transform metric names. Defaults to None.

        Returns:
            List[Any]: The filtered list of metrics that are registered in the metric registry.
        """
        validated_metrics = []

        for metric in metrics:
            metric_name = metric if name_fn is None else name_fn(metric)

            # If the metric is not registered, skip it...
            if metric_name not in metric_registry.all():
                self.logger.warning(
                    f"The metric {metric_name} is not registered in the metric registry. "
                    f"Therefore, it will not be loaded and it will not be available for the experiment. "
                    f"Check the configuration file."
                )
                continue

            # ...otherwise, load it
            validated_metrics.append(metric)

        metrics = validated_metrics
        return metrics
