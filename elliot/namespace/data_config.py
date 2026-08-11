from typing import Any, List, Optional
from pydantic import Field, model_validator, create_model

from elliot.namespace.common import BaseConfig, build_fields_from_annotations
from elliot.namespace.read_config import InteractionsReaderConfig, GeneralReaderConfig, SequenceReaderConfig
from elliot.utils.enums import DataLoadingStrategy, SessionStrategy
from elliot.utils.registry import side_info_registry
from elliot.utils import import_submodules


class SideInformationConfig(BaseConfig):
    """Base side-info configuration.

    Attributes:
        dataloader (str): Dataloader name.
        reader (GeneralReaderConfig): Reading configuration.
    """

    dataloader: str
    reader: GeneralReaderConfig = Field(default_factory=GeneralReaderConfig, exclude=True)


class DataConfig(BaseConfig):
    """Dataset loading configuration.

    Attributes:
        strategy (DataLoadingStrategy): Loading strategy to use.
        data_folder (str, optional): Path to the folder containing dataset files.
        dataset_path (str, optional): Path to the dataset file.
        sequential (bool): Whether to load sequential or interactions data. Defaults to False.
        session_strategy (SessionStrategy): Whether to segment interactions into sessions
            (SESSION_ONLY) or keep each user's whole history as a single flat sequence (FLAT).
            Segmenting requires dropping users left with fewer than two sessions, which can shrink
            some datasets substantially, so it's opt-in. Defaults to FLAT, and is always forced to
            SESSION_ONLY when `sequential` is True, since sequential data is already organized in
            per-row sessions.
        reader (Any): Reading configuration.
        side_information(List[Any]): List of side-info configurations. Defaults to [].
    """

    strategy: DataLoadingStrategy
    data_folder: Optional[str] = None
    dataset_path: Optional[str] = None
    sequential: bool = False
    session_strategy: SessionStrategy = SessionStrategy.FLAT
    reader: Any = Field(default={}, exclude=True)
    side_information: List[Any] = []

    @model_validator(mode="after")
    def resolve_session_strategy(self) -> "DataConfig":
        """Force SESSION_ONLY when reading sequential data, since each row of a
        sequential source file is already a distinct session.

        Returns:
            DataConfig: The object itself.
        """
        if self.sequential:
            self.session_strategy = SessionStrategy.SESSION_ONLY
        return self

    @model_validator(mode="after")
    def build_reader_config(self):
        reader_cls = InteractionsReaderConfig if not self.sequential else SequenceReaderConfig
        self.reader = reader_cls(**self.reader)
        return self

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
                                         f"with '{self.strategy}' strategy.")

            case DataLoadingStrategy.DATASET:
                if self.dataset_path is None:
                    raise AttributeError(f"Attribute `dataset_path` must be provided "
                                         f"with '{self.strategy}' strategy.")

        return self

    @model_validator(mode="after")
    def parse_modular_loaders(self) -> "DataConfig":
        """Parse and instantiate modular loader configurations.

        Returns:
            DataConfig: The object itself with instantiated modular loader configs.
        """
        # Import all the side-info loaders to make sure they are registered
        if self.side_information:
            import_submodules("elliot.dataset.modular_loaders")

        parsed_loaders = []

        # Handle all the side-info loaders
        for loader_data in self.side_information:
            loader_name = loader_data.get("dataloader")

            # If the loader is not registered, skip it...
            if loader_name not in side_info_registry.all():
                self.logger.warning(
                    f"The loader {loader_name} is not registered in the side info registry. "
                    f"Therefore, it will not be loaded and it will not be available for the experiment. "
                    f"Check the configuration file."
                )
                continue

            # ...otherwise, load it
            cls = side_info_registry.get_class(loader_name)

            fields = build_fields_from_annotations(cls)

            # Build loader config dynamically
            loader_config = create_model(
                f"{cls.__name__}Config",
                __base__=SideInformationConfig,
                **fields
            )

            loader_cfg = loader_config(**loader_data)
            parsed_loaders.append(loader_cfg)

        self.side_information = parsed_loaders
        return self
