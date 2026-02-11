from typing import List, Optional
from pydantic import Field, model_validator

from elliot.namespace.common import BaseConfig
from elliot.namespace.read_config import TabularReaderConfig, InteractionsReaderConfig
from elliot.utils.enums import DataLoadingStrategy


class SideInformationConfig(BaseConfig):
    """Base side-info configuration.

    Attributes:
        dataloader (str): Dataloader name.
        reader (TabularReaderConfig): Reading configuration.
    """

    dataloader: str
    reader: TabularReaderConfig = Field(default_factory=TabularReaderConfig, exclude=True)


class DataConfig(BaseConfig):
    """Dataset loading configuration.

    Attributes:
        strategy (DataLoadingStrategy): Loading strategy to use.
        data_folder (str, optional): Path to the folder containing dataset files.
        dataset_path (str, optional): Path to the dataset file.
        reader (InteractionsReaderConfig): Reading configuration.
        side_information(List[SideInformationConfig]): List of side-info configurations. Defaults to [].
    """

    strategy: DataLoadingStrategy
    data_folder: Optional[str] = None
    dataset_path: Optional[str] = None
    reader: InteractionsReaderConfig = Field(default_factory=InteractionsReaderConfig, exclude=True)
    side_information: List[SideInformationConfig] = []

    @model_validator(mode="after")
    def validate_strategy_fields(self) -> "DataConfig":
        """Validate conditional requirements based on the chosen loading strategy.

        Returns:
            DataConfig: The object itself.
        """
        match self.strategy:

            case DataLoadingStrategy.FIXED | DataLoadingStrategy.HIERARCHY:
                if self.data_folder is None:
                    raise AttributeError(f"Attribute `data_folder` must be provided "
                                         f"with '{self.strategy.value}' strategy.")

            case DataLoadingStrategy.DATASET:
                if self.dataset_path is None:
                    raise AttributeError(f"Attribute `dataset_path` must be provided "
                                         f"with '{self.strategy.value}' strategy.")

        return self
