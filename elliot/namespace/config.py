"""
Module description:

"""

__version__ = '0.3.1'

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator, create_model

from elliot.namespace.common import BaseConfig
from elliot.namespace.data_config import DataConfig
from elliot.namespace.evaluation_config import EvaluationConfig
from elliot.namespace.models_config import RecommenderConfig, MODEL_FIELD
from elliot.namespace.negative_sampling_config import NegativeSamplingConfig
from elliot.namespace.prefiltering_config import PreFilteringConfig
from elliot.namespace.results_config import ResultsConfig
from elliot.namespace.splitting_config import SplittingConfig
from elliot.utils import get_model
from elliot.utils.folder import set_config_folder, parent_dir, path_joiner, path_resolver, file_ext
from elliot.utils.hydra_config import load_config
from elliot.utils.read import Reader

reader = Reader()


class ExperimentConfig(BaseConfig):
    """Experiment configuration.

    Attributes:
        version (str): Experiment configuration version. Defaults to the current version.
        config_test (bool): Whether to use the configuration in test mode before execution. Defaults to False.
        dataset (str): Dataset name.
        data_config (DataConfig): Dataset loading configuration.
        binarize (bool): Whether to binarize interaction values. Defaults to False.
        verbose (bool): Enable verbose logging. Defaults to True.
        random_seed (int): Random seed for reproducibility. Defaults to 42.
        align_side_with_train (bool): Align side information with training split. Defaults to True.
        prefiltering (List[PreFilteringConfig]): List of pre-filtering configurations. Defaults to [].
        splitting (SplittingConfig, optional): Dataset splitting configuration.
        negative_sampling (NegativeSamplingConfig, optional): Negative sampling configuration.
        top_k (int): Number of recommended items per user. Defaults to 10.
        evaluation (EvaluationConfig): Evaluation configuration.
        results (ResultsConfig): Results handling configuration.
        gpu (int, optional): GPU device index (do not set for CPU).
        device (str, optional): Device to use (automatically picked if not set).
        torch_device (str, optional): Torch device to use (automatically picked if not set).
        backend (List[str]): List of supported training backends. Defaults to ["tensorflow"].
        path_logger_config (str): Path to the logger configuration file.
        path_log_folder (str): Path to the folder for log files.
        path_output_rec_result (str): Path to the folder for recommendation results output.
        path_output_rec_weight (str): Path to the folder for learned model weights output.
        path_output_rec_performance (str): Path to the folder for performance metrics output.
        external_models_path (str, optional): Path to the folder containing external model implementations.
        external_posthoc_path (str, optional): Path to the folder containing external post-hoc evaluators.
        models (Dict[str, Any]): Dictionary of recommender model configurations. Defaults to {}.
    """

    version: str = __version__
    config_test: bool = False
    dataset: str
    data_config: DataConfig
    binarize: bool = False
    verbose: bool = True
    random_seed: int = 42
    align_side_with_train: bool = True
    prefiltering: List[PreFilteringConfig] = []
    splitting: Optional[SplittingConfig] = None
    negative_sampling: Optional[NegativeSamplingConfig] = None
    top_k: int = 10
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    gpu: Optional[int] = None
    device: Optional[str] = None
    torch_device: Optional[str] = None
    backend: List[str] = ["tensorflow"]
    path_logger_config: str = path_joiner("..", "elliot", "config", "logger_config.yml")
    path_log_folder: str = path_joiner("..", "log")
    path_output_rec_result: str = path_joiner("..", "results", "{0}", "recs")
    path_output_rec_weight: str = path_joiner("..", "results", "{0}", "weights")
    path_output_rec_performance: str = path_joiner("..", "results", "{0}", "performance")
    external_models_path: Optional[str] = None
    external_posthoc_path: Optional[str] = None
    models: Dict[str, Any] = {}

    @model_validator(mode="after")
    def resolve_paths(self) -> "ExperimentConfig":
        """Resolve dataset-dependent paths inside the configuration.

        Returns:
            ExperimentConfig: The object itself with resolved paths.
        """
        excluded = {"reader", "writer", "version"}

        def _resolve(obj: Any) -> Any:
            if isinstance(obj, str):
                return path_resolver(obj, self.dataset)
            if isinstance(obj, list):
                return [_resolve(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _resolve(v) if k not in excluded else v for k, v in obj.items()}
            if isinstance(obj, BaseModel):
                all_fields = set(obj.model_fields.keys()) | set((obj.model_extra or {}).keys())
                for name in all_fields:
                    value = getattr(obj, name)
                    if value is not None and name not in excluded:
                        setattr(obj, name, _resolve(value))
            return obj

        _resolve(self)
        return self

    @model_validator(mode="after")
    def handle_evaluation_metrics_and_cutoffs(self) -> "ExperimentConfig":
        """Adjust evaluation metrics and cutoff values after model validation.

        Returns:
            ExperimentConfig: The object itself with modified evaluation settings.
        """
        cutoff_k = self.evaluation.cutoffs or [self.top_k]
        self.evaluation.cutoffs = cutoff_k
        self.results.default_k = cutoff_k[0]

        first_metric = (
            self.evaluation.simple_metrics[0]
            if self.evaluation.simple_metrics else ""
        )
        self.results.default_metric = first_metric

        return self

    @model_validator(mode="after")
    def parse_models(self) -> "ExperimentConfig":
        """Parse and instantiate recommender model configurations.

        Returns:
            ExperimentConfig: The object itself with instantiated model configs.
        """

        # Handle 'RecommendationFolder'...
        if "RecommendationFolder" in self.models:
            self.handle_recommendation_folder()

        # ...and all the other models
        for model_name, model_data in self.models.items():
            cls = get_model(model_name, self)

            fields = self._build_fields_from_annotations(cls)

            # Build recommender config dynamically
            model_config = create_model(
                f"{cls.__name__}Config",
                __base__=RecommenderConfig,
                **fields
            )

            self.models[model_name] = model_config(**model_data)

        return self

    def handle_recommendation_folder(self):
        """Expand a folder-based recommender configuration into multiple proxy models.

        This method processes a special meta-model named "RecommendationFolder".
        Each file found in the specified folder is converted into a separate
        proxy recommender configuration, inheriting the original settings and
        overriding the input path.

        Raises:
            AttributeError: If the required 'folder' field is missing.
        """
        recommender_name = "RecommendationFolder"
        model_data = self.models[recommender_name]

        folder_path = model_data.get("folder")
        if folder_path is None:
            raise AttributeError(f"{recommender_name} meta-model must expose the `folder` field.")

        files = reader.read_folder(
            folder=folder_path,
            patterns=model_data.get("patterns"),
            ext=model_data.get("ext")
        )

        # Create one 'ProxyRecommender' configuration for each file
        for i, file_ in enumerate(files):
            single_data = {k: v for k, v in model_data.items() if k != "folder"}
            single_data["path"] = file_
            single_data.setdefault("reader", {})["ext"] = file_ext(file_)
            self.models[f"ProxyRecommender{i+1}"] = single_data

        # Remove 'RecommendationFolder' entry
        self.models.pop(recommender_name)

    def _build_fields_from_annotations(self, cls: object) -> dict:
        """Build Pydantic field definitions from class annotations.

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

            fields[name] = (MODEL_FIELD(hint), default)

        return fields


def build_namespace(
    config_path: str,
    config_overrides: Optional[List[str]] = None,
    config_data: Optional[dict] = None
) -> ExperimentConfig:
    """Build the experiment namespace from configuration sources.

    Args:
        config_path (str): Path to the main configuration file.
        config_overrides (List[str], optional): Optional override directives.
        config_data (dict, optional): Optional pre-loaded configuration data.

    Returns:
        ExperimentConfig: Fully initialized experiment configuration.
    """
    set_config_folder(parent_dir(config_path))

    if config_data is not None:
        config = config_data
    else:
        config = load_config(config_path, overrides=config_overrides)

    return ExperimentConfig(**config["experiment"])
