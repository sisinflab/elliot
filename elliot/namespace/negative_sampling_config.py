from typing import Optional
from pydantic import Field, model_validator

from elliot.namespace.common import BaseConfig
from elliot.namespace.read_config import TabularReaderConfig
from elliot.namespace.write_config import TabularWriterConfig
from elliot.utils.enums import NegativeSamplingStrategy
from elliot.utils.folder import path_joiner


class NegativeSamplingConfig(BaseConfig):
    """Negative sampling configuration.

    Attributes:
        strategy (NegativeSamplingStrategy): Negative sampling strategy to use.
        num_negatives (int): Number of negative samples. Defaults to 99, min is 1.
        leave_one_out (bool): Whether to add only one positive item to the sampled negatives
            per user. Defaults to False.
        save_on_disk (bool): Whether to save sampling results to disk. Defaults to False.
        save_folder (str, optional): Path to the folder where negative samples files will be saved.
        read_folder (str, optional): Path to the folder containing negative samples files;
            required for `fixed` strategy.
        reader (TabularReaderConfig): Reading configuration.
        writer (TabularWriterConfig): Writing configuration.
    """

    strategy: NegativeSamplingStrategy
    num_negatives: int = Field(default=99, ge=1)
    leave_one_out: bool = False
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
