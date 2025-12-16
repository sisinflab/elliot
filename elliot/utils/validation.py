from typing import get_origin, Any, Union, Optional
from types import SimpleNamespace
import ast
from pydantic import BaseModel, Field, model_validator, create_model, GetCoreSchemaHandler, ConfigDict
from pydantic.v1.main import ModelMetaclass
from pydantic_core import core_schema

from elliot.utils.enums import PreFilteringStrategy, SplittingStrategy, NegativeSamplingStrategy, DataLoadingStrategy


class BaseValidator(BaseModel):
    """Base validator with utility to get the validated params dict."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_validated_params(self, *args) -> dict:
        """Return all the validated fields.

        Returns:
            dict: The validated params dict.
        """
        return self.model_dump()


# DataSetLoader configuration

class DataLoadingConfig(BaseValidator):
    """Dataset loading configuration.

    Attributes:
        strategy (DataLoadingStrategy): Loading strategy to use.
        dataset_path (Optional[str]): Path to the full dataset.
        root_folder (Optional[str]): Root directory containing dataset files.
        train_path (Optional[str]): Path to the training dataset.
        validation_path (Optional[str]): Path to the validation dataset.
        test_path (Optional[str]): Path to the test dataset.
        header (bool): Whether the dataset(s) include(s) a header row; default is False.
        binarize (bool): Whether to binarize the dataset; default is False.
        seed (int): Random seed; default is 42.
    """

    strategy: DataLoadingStrategy
    dataset_path: Optional[str] = Field(default=None)
    root_folder: Optional[str] = Field(default=None)
    train_path: Optional[str] = Field(default=None)
    validation_path: Optional[str] = Field(default=None)
    test_path: Optional[str] = Field(default=None)
    header: bool = Field(default=False)
    binarize: bool = Field(default=False)
    seed: int = Field(default=42)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "DataLoadingConfig":
        """Validate conditional requirements based on the chosen loading strategy.

        Returns:
            DataSetLoadingConfig: The object itself.
        """
        match self.strategy:

            case DataLoadingStrategy.FIXED:
                if self.train_path is None or self.test_path is None:
                    raise AttributeError(f"Both `train_path` and `test_path` must be provided "
                                         f"with `{self.strategy.value}` strategy.")

            case DataLoadingStrategy.HIERARCHY:
                if self.root_folder is None:
                    raise AttributeError(f"Attribute `root_folder` must be provided "
                                         f"with `{self.strategy.value}` strategy.")

            case DataLoadingStrategy.DATASET:
                if self.dataset_path is None:
                    raise AttributeError(f"Attribute `dataset_path` must be provided "
                                         f"with `{self.strategy.value}` strategy.")

        return self


# PreFilter validation

class PreFilteringValidator(BaseValidator):
    """Pre-filtering validator.

    Attributes:
        strategy (PreFilteringStrategy): Pre-filtering strategy to use.
        threshold (Optional[Union[float, int]]): Threshold value for filtering; min is 0.
        core (int): Core parameter for the strategy; default is 5.
        rounds (int): Number of rounds to perform; default is 2.
    """

    strategy: PreFilteringStrategy
    threshold: Optional[Union[float, int]] = Field(default=None, ge=0)
    core: int = Field(default=5, ge=0)
    rounds: int = Field(default=2, ge=0)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "PreFilteringValidator":
        """Ensure required fields are set for the selected pre-filtering strategy.

        Returns:
            PreFilteringValidator: The object itself.
        """
        if self.strategy == PreFilteringStrategy.COLD_USERS and self.threshold is None:
            raise AttributeError(f"Attribute `threshold` must be provided "
                                 f"with `{self.strategy.value}` strategy.")

        return self


# Splitter validation

class SplittingGeneralValidator(BaseValidator):
    """Splitting general validator.

    Attributes:
        save_on_disk (bool): Whether to save split data to disk; default is False.
        save_folder (Optional[str]): Folder path to save splits if `save_on_disk` is True.
    """

    save_on_disk: bool = Field(default=False)
    save_folder: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_general_fields(self) -> "SplittingGeneralValidator":
        """Ensure required fields are set if saving splits to disk.

        Returns:
            SplittingGeneralValidator: The object itself.
        """
        if self.save_on_disk and self.save_folder is None:
            raise AttributeError("Attribute `save_folder` must be provided if `save_on_disk` is set to True.")

        return self


class SplittingValidator(BaseValidator):
    """Splitting validator.

    Attributes:
        strategy (SplittingStrategy): Splitting strategy to apply.
        timestamp (Optional[float]): Optional timestamp for splitting.
        min_below (int): Minimum number of items below threshold; default is 1.
        min_over (int): Minimum number of items over threshold; default is 1.
        test_ratio (Optional[float]): Fraction of data for testing; min is 0.1, max is 0.9.
        leave_n_out (Optional[int]): Number of items to leave out for test.
        folds (int): Number of folds for cross-validation; default is 5, min is 1, max is 20.
    """

    strategy: SplittingStrategy = Field()
    timestamp: Optional[float] = Field(default=None)
    min_below: int = Field(default=1, ge=1)
    min_over: int = Field(default=1, ge=1)
    test_ratio: Optional[float] = Field(default=None, ge=0.1, le=0.9)
    leave_n_out: Optional[int] = Field(default=None, ge=1)
    folds: int = Field(default=5, ge=1, le=20)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "SplittingValidator":
        """Validate conditional requirements based on the chosen splitting strategy.

        Returns:
            SplittingValidator: The object itself.
        """
        match self.strategy:

            case SplittingStrategy.FIXED_TS:
                pass

            case SplittingStrategy.TEMP_HOLDOUT:
                if self.test_ratio is None and self.leave_n_out is None:
                    raise AttributeError(f"At least one among `test_ratio` and `leave_n_out` must be provided "
                                         f"with `{self.strategy.value}` strategy.")

            case SplittingStrategy.RAND_SUB_SMP:
                if self.test_ratio is None and self.leave_n_out is None:
                    raise AttributeError(f"At least one among `test_ratio` and `leave_n_out` must be provided "
                                         f"with `{self.strategy.value}` strategy.")

            case SplittingStrategy.RAND_CV:
                min_val = 2
                if self.folds < min_val:
                    raise ValueError(f"Attribute `folds` must be at least {min_val} "
                                     f"with `{self.strategy.value}` strategy.")

        return self


# NegativeSampler validation

class NegativeSamplingValidator(BaseValidator):
    """Negative sampling validator.

    Attributes:
        strategy (NegativeSamplingStrategy): Negative sampling strategy to use.
        num_negatives (int): Number of negative samples; default is 99, min is 1.
        save_on_disk (bool): Whether to save sampling results to disk; default is False.
        file_path (Optional[str]): File path for saving negative samples.
        test_file_path (Optional[str]): File path for test negative samples; required for FIXED strategy.
        val_file_path (Optional[str]): File path for validation negative samples.
    """

    strategy: NegativeSamplingStrategy
    num_negatives: int = Field(default=99, ge=1)
    save_on_disk: bool = Field(default=False)
    file_path: Optional[str] = Field(default=None)
    test_file_path: Optional[str] = Field(default=None)
    val_file_path: Optional[str] = Field(default=None)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "NegativeSamplingValidator":
        """Ensure required fields are set for the selected negative sampling strategy.

        Returns:
            NegativeSamplingValidator: The object itself.
        """
        if self.strategy == NegativeSamplingStrategy.FIXED and self.test_file_path is None:
            raise AttributeError(f"Attribute `test_file_path` must be provided "
                                 f"with `{self.strategy.value}` strategy.")

        return self


# Trainer validation

class TrainerValidator(BaseValidator):
    """Training validator.

    Attributes:
        restore (bool): Whether to restore a previous training state; default is False.
        validation_metric (Union[str, list]): Validation metric(s) to compute.
        save_weights (bool): Whether to save model weights after training; default is False.
        save_recs (bool): Whether to save generated recommendations; default is False.
        verbose (bool): Whether to enable verbose logging; default is True.
        validation_rate (int): Frequency (in epochs) of validation runs; default is 1, min is 1.
        optimize_internal_loss (bool): Whether to optimize internal loss instead of main objective; default is False.
        epochs (int): Number of training epochs; default is 1, min is 1.
        batch_size (int): Training batch size; min is 1.
        eval_batch_size(int): Evaluation batch size; min is 1.
        seed (int): Random seed; default is 42.
    """

    restore: bool = Field(default=False)
    validation_metric: Union[str, list] = Field(default=None)
    save_weights: bool = Field(default=False)
    save_recs: bool = Field(default=False)
    verbose: bool = Field(default=True)
    validation_rate: int = Field(default=1, ge=1)
    optimize_internal_loss: bool = Field(default=False)
    epochs: int = Field(default=1, ge=1)
    batch_size: int = Field(default=None, ge=1)
    eval_batch_size: int = Field(default=None, ge=1)
    seed: int = Field(default=42)

    def get_validated_params(self, meta: bool = False, *args):
        """Return a subset of the validated fields.

        Args:
            meta (bool): If True assign metadata fields, otherwise training fields.
        """

        out_of_meta = ['epochs', 'batch_size', 'eval_batch_size', 'seed']

        # Pick the right fields to retrieve
        included = set(out_of_meta) if not meta else set(self.model_fields.keys()) - set(out_of_meta)
        validated_data = self.model_dump(include=included)

        return validated_data


# Recommender validation

class TupleFromString:
    """Custom type converting tuple-like strings into real tuples.

    Accepts strings, lists, and tuples.
    """

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: GetCoreSchemaHandler):
        """Return Pydantic Core schema applying the custom validation.

        Args:
            source_type (Any): Only for compatibility.
            handler (GetCoreSchemaHandler): Only for compatibility.
        """
        return core_schema.no_info_after_validator_function(
            cls.validate,
            core_schema.any_schema()
        )

    @staticmethod
    def validate(value: Union[str, list, tuple]):
        """Convert input into a tuple, parsing strings with literal_eval.

        Args:
            value (Union[str, list, tuple]): The value to be cast to tuple.

        Returns:
            tuple: The cast value.
        """
        if isinstance(value, str):
            value = ast.literal_eval(value)
        return tuple(value)


def build_recommender_validator(cls: object) -> ModelMetaclass:
    """Create a dynamic Pydantic validator for a recommender class.

    Tuple fields are automatically converted using TupleFromString.

    Args:
        cls (object): Abstract class of the recommender.

    Returns:
        ModelMetaclass: The created Pydantic validator.
    """

    fields = _build_fields_from_annotations(cls)

    # Build recommender validator dynamically
    dynamic_validator = create_model(
        f"{cls.__name__}Validator",
        __base__=BaseValidator,
        **fields
    )

    return dynamic_validator


def _build_fields_from_annotations(cls: object) -> dict:
    """Build Pydantic field definitions from class annotations.

    Tuple fields are replaced with TupleFromString for string parsing.

    Args:
        cls (object): The class from which keeping the annotations.

    Returns:
        dict: The extracted fields' dict.
    """
    fields = {}

    for name, hint in cls.__annotations__.items():
        # Skip 'type' attribute
        # (used only to pick the right trainer)
        if name == "type":
            continue

        # Get default value
        default = getattr(cls, name)

        # Use custom type 'TupleFromString'
        # to automatically cast str to tuple
        if get_origin(hint) is tuple:
            hint = TupleFromString

        fields[name] = (hint, default)

    return fields


# Further validation utils

def check_range(
    attr_name: str,
    attr_val: Union[int, float],
    min_val: Union[int, float],
    max_val: Union[int, float]
):
    """Check if a numeric attribute value is within a specified range.

    Args:
        attr_name (str): Name of the attribute being validated.
        attr_val (Union[int, float]): Value of the attribute to check.
        min_val (Union[int, float]): Minimum allowed value (inclusive).
        max_val (Union[int, float]): Maximum allowed value (inclusive).

    Raises:
        ValueError: If `attr_val` is not within the range [min_val, max_val].
    """
    if not min_val <= attr_val <= max_val:
        raise ValueError(f"Attribute `{attr_name}` must be between {min_val} and {max_val} "
                         f"for the provided dataset.")
