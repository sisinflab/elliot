from typing import List, Any, Union, Optional
from ast import literal_eval
from pydantic import Field, model_validator, field_validator
from pydantic_core.core_schema import ValidationInfo

from elliot.namespace.common import BaseConfig, check_type
from elliot.namespace.read_config import InteractionsReaderConfig, ModelReaderConfig
from elliot.namespace.write_config import TabularWriterConfig, ModelWriterConfig
from elliot.utils.enums import SearchSpace, OptimizationAlgorithm, SessionStrategy

MODEL_FIELD = lambda t: Union[t, List[Union[str, t]]]


class MetaConfig(BaseConfig):
    """Meta configuration.

    Attributes:
        restore (bool): Whether to restore a previous training state. Defaults to False.
        save_weights (bool): Whether to save model weights after training. Defaults to False.
        save_recs (bool): Whether to save generated recommendations. Defaults to False.
        verbose (bool): Enable verbose logging. Defaults to True.
        validation_metric (str): Metric used for validation (automatically set if not provided).
            Defaults to None.
        validation_rate (int): Frequency (in epochs) of validation runs. Defaults to 1, min is 1.
        optimization_target (str, optional): Target metric or loss to optimize.
        optimize_internal_loss (bool): Whether to optimize the internal loss. Defaults to False.
        hyper_max_evals (int, optional): Maximum number of hyperparameter evaluations.
        hyper_opt_alg (OptimizationAlgorithm): Hyperparameter optimization algorithm. Defaults to "tpe".
        session_strategy (SessionStrategy): Strategy for session training and evaluation. Defaults to "flat".
        model_reader (ModelReaderConfig): Model reading configuration.
        model_writer (ModelWriterConfig): Model writing configuration.
        rec_reader (InteractionsReaderConfig): Recommendation reading configuration.
        rec_writer (TabularWriterConfig): Recommendation writing configuration.
    """

    restore: bool = False
    save_weights: bool = False
    save_recs: bool = False
    verbose: bool = True
    validation_metric: str = ""
    validation_rate: int = Field(default=1, ge=1)
    optimization_target: str = None
    optimize_internal_loss: bool = False
    hyper_max_evals: Optional[int] = None
    hyper_opt_alg: OptimizationAlgorithm = OptimizationAlgorithm.TPE
    session_strategy: SessionStrategy = SessionStrategy.FLAT
    model_reader: ModelReaderConfig = Field(default_factory=ModelReaderConfig, exclude=True)
    model_writer: ModelWriterConfig = Field(default_factory=ModelWriterConfig, exclude=True)
    rec_reader: InteractionsReaderConfig = Field(default_factory=InteractionsReaderConfig, exclude=True)
    rec_writer: TabularWriterConfig = Field(default_factory=TabularWriterConfig, exclude=True)

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
        monitor (str): Metric or quantity to monitor. Defaults to "".
        patience (int): Number of epochs with no improvement before stopping. Defaults to 0.
        mode (str, optional): Optimization mode ("min" or "max"). Defaults to None.
        min_delta (int, optional): Minimum absolute change to qualify as an improvement. Defaults to None.
        rel_delta (int, optional): Minimum relative change to qualify as an improvement. Defaults to None.
        baseline (int, optional): Baseline value for the monitored quantity. Defaults to None.
        verbose (bool): Enable verbose logging. Defaults to True.
    """

    monitor: str = ""
    patience: int = Field(default=0, ge=0)
    mode: Optional[str] = None
    min_delta: Optional[float] = Field(default=None, ge=0)
    rel_delta: Optional[float] = Field(default=None, ge=0)
    baseline: Optional[float] = Field(default=None, ge=0)
    verbose: bool = True

    @field_validator("mode", mode="before")
    @classmethod
    def check_mode(cls, value: Any) -> Any:
        """Validate the "mode" configuration field.

        Args:
            value (Any): Raw field value from the configuration.

        Returns:
            Any: Validated field value.

        Raises:
            ValueError: If the value of "mode" is not None and not one of ['min', 'max', 'auto'].
        """
        allowed = ["min", "max", "auto"]

        if value is not None and value not in allowed:
            raise ValueError(f"Attribute `mode` must be one of {allowed}.")

        return value


class RecommenderConfig(BaseConfig):
    """Base recommender configuration.

    Attributes:
        meta (MetaConfig): Meta-level training configuration.
        early_stopping (Optional[EarlyStoppingConfig]): Early stopping configuration.
        epochs (MODEL_FIELD(int)): Number of training epochs (or search space). Defaults to 1.
        batch_size (MODEL_FIELD(int)): Training batch size (or search space). Defaults to 1024.
        eval_batch_size (MODEL_FIELD(int)): Evaluation batch size (or search space). Defaults to None.
        best_iteration (int): Best epoch selected during training (automatically set).
        name (str): Recommender instance name (automatically set).
    """

    meta: MetaConfig = Field(default_factory=MetaConfig)
    early_stopping: Optional[EarlyStoppingConfig] = Field(default=None, exclude=True)
    epochs: MODEL_FIELD(int) = 1
    batch_size: MODEL_FIELD(int) = 1024
    eval_batch_size: MODEL_FIELD(int) = None
    best_iteration: int = None
    name: str = None

    warn_on_extra_fields = True

    @field_validator("*", mode="before")
    @classmethod
    def eval_tuple_fields(cls, value: Any, info: ValidationInfo) -> Any:
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
