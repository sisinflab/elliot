from typing import Optional
from pydantic import Field, model_validator

from elliot.namespace.common import BaseConfig
from elliot.namespace.read_write_config import WriterConfig
from elliot.utils.enums import SplittingStrategy
from elliot.utils.folder import path_joiner


class SplittingSingleConfig(BaseConfig):
    """Splitting configuration.

    Attributes:
        strategy (SplittingStrategy): Splitting strategy to apply.
        timestamp (Optional[float]): Optional timestamp for splitting.
        min_below (int): Minimum number of items below threshold; default is 1, min is 1.
        min_over (int): Minimum number of items over threshold; default is 1, min is 1.
        test_ratio (Optional[float]): Fraction of data for testing; min is 0.1, max is 0.9.
        leave_n_out (Optional[int]): Number of items to leave out for test.
        folds (int): Number of folds for cross-validation; default is 5, min is 1, max is 20.
    """

    strategy: SplittingStrategy
    timestamp: Optional[float] = None
    min_below: int = Field(default=1, ge=1)
    min_over: int = Field(default=1, ge=1)
    test_ratio: Optional[float] = Field(default=None, ge=0.1, le=0.9)
    leave_n_out: Optional[int] = Field(default=None, ge=1)
    folds: int = Field(default=1, ge=1, le=20)

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "SplittingSingleConfig":
        """Validate conditional requirements based on the chosen splitting strategy.

        Returns:
            SplittingSingleConfig: The object itself.
        """
        match self.strategy:

            case SplittingStrategy.FIXED_TS:
                pass

            case SplittingStrategy.TEMP_HOLDOUT:
                if self.test_ratio is None and self.leave_n_out is None:
                    raise AttributeError(f"At least one among `test_ratio` and `leave_n_out` must be provided "
                                         f"with '{self.strategy.value}' strategy.")

            case SplittingStrategy.RAND_SUB_SMP:
                if self.test_ratio is None and self.leave_n_out is None:
                    raise AttributeError(f"At least one among `test_ratio` and `leave_n_out` must be provided "
                                         f"with '{self.strategy.value}' strategy.")

            case SplittingStrategy.RAND_CV:
                min_val = 2
                if self.folds < min_val:
                    raise ValueError(f"Attribute `folds` must be at least {min_val} "
                                     f"with '{self.strategy.value}' strategy.")

        return self


class SplittingConfig(BaseConfig):
    """Splitting general configuration.

    Attributes:
        save_on_disk (bool): Whether to save split data to disk; default is False.
        save_folder (str): Folder path to save splits if `save_on_disk` is True.
        test_splitting (SplittingSingleConfig): Test splitting configuration.
        validation_splitting (Optional[SplittingSingleConfig]): Validation splitting configuration.
        writer (WriterConfig): Writing configuration.
    """

    save_on_disk: bool = False
    save_folder: str = path_joiner("..", "data", "{0}", "splitting")
    test_splitting: SplittingSingleConfig
    validation_splitting: Optional[SplittingSingleConfig] = None
    writer: WriterConfig = Field(default_factory=WriterConfig, exclude=True)
