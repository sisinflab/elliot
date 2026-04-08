"""
Module description:

"""

__version__ = '0.3.1'

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, model_validator, create_model

from elliot.namespace.common import BaseConfig, build_fields_from_annotations
from elliot.namespace.data_config import DataConfig
from elliot.namespace.evaluation_config import EvaluationConfig
from elliot.namespace.models_config import RecommenderConfig, MODEL_FIELD
from elliot.namespace.negative_sampling_config import NegativeSamplingConfig
from elliot.namespace.prefiltering_config import PreFilteringConfig
from elliot.namespace.results_config import ResultsConfig
from elliot.namespace.splitting_config import SplittingConfig
from elliot.namespace.wandb_config import WandBConfig
from elliot.utils import split_metric, import_submodules
from elliot.utils.folder import set_config_folder, parent_dir, path_joiner, path_resolver, file_ext
from elliot.utils.hydra_config import load_config
from elliot.utils.read import Reader
from elliot.utils.registry import model_registry

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
    wandb: Optional[WandBConfig] = None
    top_k: int = 10
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    results: ResultsConfig = Field(default_factory=ResultsConfig)
    gpu: Optional[int] = None
    device: Optional[str] = None
    torch_device: Optional[str] = None
    backend: List[str] = ["tensorflow"]
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
        excluded = {
            "reader", "rec_reader", "model_reader",
            "writer", "rec_writer", "model_writer", "version"
        }

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

        first_metric = (
            self.evaluation.simple_metrics[0]
            if self.evaluation.simple_metrics else ""
        )
        self.results.default_metric = first_metric + "@" + str(cutoff_k[0])

        return self

    @model_validator(mode="after")
    def parse_models(self) -> "ExperimentConfig":
        """Parse and instantiate recommender model configurations.

        Returns:
            ExperimentConfig: The object itself with instantiated model configs.
        """
        # Import all the models and the samplers to make sure they are registered
        if self.models:
            import_submodules("elliot.recommender")
            import_submodules("elliot.dataset.samplers")

        # Handle 'RecommendationFolder'...
        if "RecommendationFolder" in self.models:
            self.handle_recommendation_folder()

        parsed_models = {}

        # ...and all the other models
        for model_name, model_data in self.models.items():
            is_proxy = model_name.startswith("Proxy")
            if is_proxy:
                cls_name = "ProxyRecommender"
                field_fn = None
            else:
                cls_name = model_name
                field_fn = MODEL_FIELD

            # If the model is not registered, skip it...
            if cls_name not in model_registry.all():
                self.logger.warning(
                    f"The model {cls_name} is not registered in the model registry. "
                    f"Therefore, it will not be loaded and it will not be available for the experiment. "
                    f"Check the configuration file."
                )
                continue

            # ...otherwise, load it
            cls = model_registry.get_class(cls_name)

            fields = build_fields_from_annotations(cls, field_fn=field_fn)

            # Build recommender config dynamically
            model_config = create_model(
                f"{cls.__name__}Config",
                __base__=RecommenderConfig,
                **fields
            )

            model_cfg = model_config(**model_data)

            # Handle validation metric
            metric = self._check_metric(
                metric=model_cfg.meta.validation_metric,
                default=self.results.default_metric
            )
            model_cfg.meta.validation_metric = metric

            # Handle early stopping metric
            if model_cfg.early_stopping is not None:
                metric = self._check_metric(
                    metric=model_cfg.early_stopping.monitor,
                    default=model_cfg.meta.validation_metric
                )
                model_cfg.early_stopping.monitor = metric

            parsed_models[model_name] = model_cfg

        self.models = parsed_models
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

        rec_reader = model_data.get("rec_reader", {})

        files = reader.read_folder(
            folder=folder_path,
            patterns=rec_reader.get("patterns"),
            ext=rec_reader.get("ext")
        )

        # Create one 'ProxyRecommender' configuration for each file
        for i, file_ in enumerate(files):
            single_data = {k: v for k, v in model_data.items() if k != "folder" and k != "rec_reader"}
            single_data["path"] = file_
            single_data["meta"] = {"rec_reader": {"ext": file_ext(file_)}}
            self.models[f"ProxyRecommender{i+1}"] = single_data

        # Remove 'RecommendationFolder' entry
        self.models.pop(recommender_name)

    def _check_metric(self, metric: str, default: str) -> str:
        """Check and validate a metric name and cutoff value combination.

        Args:
            metric (str): Metric string to validate. Can be a combination of metric name
                and cutoff value (e.g., 'precision@5').
            default (str): Default metric string to use if parts of the given metric are missing.

        Returns:
            str: Final validated metric string in the format 'metric_name@cutoff_value'.

        Raises:
            ValueError: If the metric name is not one of the simple metrics in `self.evaluation.simple_metrics`,
                or if the cutoff value is not one of the cutoffs in `self.evaluation.cutoffs`.
        """
        metric_name, top_k = split_metric(metric)
        default_name, default_k = split_metric(default)

        # Pick the metric and cutoff value to use
        final_name = metric_name or default_name
        final_k = top_k or default_k

        if final_name.lower() not in [m.lower() for m in self.evaluation.simple_metrics]:
            raise ValueError(f"Metric '{final_name}' is not in the list of simple metrics.")

        if final_k not in self.evaluation.cutoffs:
            raise ValueError(f"Cutoff {final_k} is not in general cutoff values.")

        return final_name + "@" + str(final_k)


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

    # Import external folder to register all the custom components
    # import_submodules("external")

    if config_data is not None:
        config = config_data
    else:
        config = load_config(config_path, overrides=config_overrides)

    return ExperimentConfig(**config["experiment"])
