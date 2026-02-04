from typing import List, Any, Union, Optional
from ast import literal_eval
from pydantic import Field, model_validator, field_validator
from pydantic_core.core_schema import ValidationInfo

from elliot.namespace.common import BaseConfig, check_type
from elliot.namespace.read_write_config import ReaderConfig, WriterConfig
from elliot.utils.enums import SearchSpace, OptimizationAlgorithm

MODEL_FIELD = lambda t: Union[t, List[Union[str, t]]]


class MetaConfig(BaseConfig):
    """Meta configuration.

    Attributes:
        restore (bool): Whether to restore a previous training state; default is False.
        save_weights (bool): Whether to save model weights after training; default is False.
        save_recs (bool): Whether to save generated recommendations; default is False.
        verbose (bool): Enable verbose logging; default is True.
        validation_metric (Optional[str]): Metric used for validation.
        validation_k (Optional[int]): Cutoff value for validation metrics (automatically set).
        validation_rate (int): Frequency (in epochs) of validation runs; default is 1, min is 1.
        optimization_target (Optional[str]): Target metric or loss to optimize.
        optimize_internal_loss (bool): Whether to optimize the internal loss; default is False.
        hyper_max_evals (Optional[int]): Maximum number of hyperparameter evaluations.
        hyper_opt_alg (OptimizationAlgorithm): Hyperparameter optimization algorithm; default is "tpe".
    """

    restore: bool = False
    save_weights: bool = False
    save_recs: bool = False
    verbose: bool = True
    validation_metric: str = None
    validation_k: int = None
    validation_rate: int = Field(default=1, ge=1)
    optimization_target: str = None
    optimize_internal_loss: bool = False
    hyper_max_evals: Optional[int] = None
    hyper_opt_alg: OptimizationAlgorithm = OptimizationAlgorithm.TPE

    @model_validator(mode="after")
    def validate_optimization_target(self) -> "MetaConfig":
        """Validate and initialize the optimization target.

        Returns:
            MetaConfig: The object itself.
        """
        if self.optimization_target is None and self.optimize_internal_loss:
            self.optimization_target = "internal_loss"
        self.optimization_target = self.optimization_target or "validation_metric"
        return self


class EarlyStoppingConfig(BaseConfig):
    """Early stopping configuration.

    Attributes:
        monitor (str): Metric or quantity to monitor; default is "".
        patience (int): Number of epochs with no improvement before stopping; default is 0.
        mode (Optional[str]): Optimization mode ("min" or "max").
        min_delta (Optional[int]): Minimum absolute change to qualify as improvement.
        rel_delta (Optional[int]): Minimum relative change to qualify as improvement.
        baseline (Optional[int]): Baseline value for the monitored quantity.
        verbose (bool): Enable verbose logging; default is True.
    """

    monitor: str = ""
    patience: int = 0
    mode: Optional[str] = None
    min_delta: Optional[int] = None
    rel_delta: Optional[int] = None
    baseline: Optional[int] = None
    verbose: bool = True


class RecommenderConfig(BaseConfig):
    """Base recommender configuration.

    Attributes:
        meta (MetaConfig): Meta-level training configuration.
        early_stopping (Optional[EarlyStoppingConfig]): Early stopping configuration.
        epochs (MODEL_FIELD(int)): Number of training epochs (or search space); default is 1.
        batch_size (MODEL_FIELD(int)): Training batch size (or search space); default is 1024.
        eval_batch_size (MODEL_FIELD(int)): Evaluation batch size (or search space).
        best_iteration (int): Best epoch selected during training (automatically set).
        name (str): Recommender instance name (automatically set).
    """

    meta: MetaConfig = Field(default_factory=MetaConfig, exclude=True)
    early_stopping: Optional[EarlyStoppingConfig] = Field(default=None, exclude=True)
    epochs: MODEL_FIELD(int) = 1
    batch_size: MODEL_FIELD(int) = 1024
    eval_batch_size: MODEL_FIELD(int) = None
    best_iteration: int = 0
    name: str = ""
    reader: ReaderConfig = Field(default_factory=ReaderConfig, exclude=True)
    writer: WriterConfig = Field(default_factory=WriterConfig, exclude=True)

    @field_validator("*", mode="before")
    @classmethod
    def eval_tuple_fields(cls, value: Any, info: ValidationInfo):
        """Parse tuple-like configuration fields.

        Args:
            value (Any): Raw field value from the configuration.
            info (ValidationInfo): Pydantic field metadata.

        Returns:
            Any: Parsed and validated field value.
        """
        field = cls.model_fields[info.field_name]
        hint = field.annotation

        if check_type(hint, tuple):
            if isinstance(value, str):
                value = literal_eval(value)
            elif isinstance(value, list):
                search, parse = (value[0], value[1:]) if value[0] in SearchSpace else ([], value)
                parse = [literal_eval(t) for t in parse]
                value = search + parse

        return value

    def prepare_fields_for_search(self):
        """Prepare configuration fields for hyperparameter search.

        Convert scalar values into search space definitions when
        required and ensure compatibility with optimization engines.
        """
        for name, field in self.model_fields.items():
            hint = field.annotation
            value = getattr(self, name)

            if check_type(hint, list):
                if not isinstance(value, list):
                    value = [value]
                if not value[0] in SearchSpace:
                    value = [SearchSpace.CHOICE.value] + value

            setattr(self, name, value)
