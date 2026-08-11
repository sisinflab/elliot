from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from elliot.dataset import DataSet
    from elliot.namespace import RecommenderConfig
    from elliot.recommender import AbstractRecommender

SPLITTING_TYPE = List[Tuple[List[Tuple[pd.DataFrame, pd.DataFrame]], Optional[pd.DataFrame], pd.DataFrame]]


class ElliotCallback:
    """Base class for Elliot callbacks.

    This class provides a base for custom Elliot callbacks.
    Custom callbacks should inherit from this class and implement the necessary methods.

    Args:
        *args (Any): Additional positional arguments.
        **kwargs (Any): Additional keyword arguments.
    """

    def __init__(self, *args: Any, **kwargs: Any):
        pass

    def on_data_loading_and_filtering(
        self,
        data: Union[pd.DataFrame, SPLITTING_TYPE],
        *args: Any,
        **kwargs: Any,
    ):
        """Callback method to be called after the data loading and filtering.

        Args:
            data (Union[pd.DataFrame, SPLITTING_TYPE]): The data that has been read.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_dataset_creation(
        self,
        main_dataset: List["DataSet"],
        val_dataset: List[List["DataSet"]],
        *args: Any,
        **kwargs: Any,
    ):
        """Callback method to be called after dataset creation.

        Args:
            main_dataset (Dataset): The main dataset that has been created.
                Contains information about the train/test split, for each test fold.
            val_dataset (Dataset): The validation dataset that has been created.
                Contains information about the train/val split, for each test fold.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_model_start(
        self,
        model_name: str,
        model_config: "RecommenderConfig",
        *args: Any,
        **kwargs: Any
    ):
        """Callback method to be called before training and testing starts (for a specific model).

        Args:
            model_name (str): The name of the model.
            model_config (RecommenderConfig): The configuration of the model.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_training_complete(
        self,
        model: "AbstractRecommender",
        results: Dict[str, Any],
        *args: Any,
        **kwargs: Any
    ):
        """Callback method to be called after training is complete (for a specific test fold).

        Args:
            model (AbstractRecommender): The trained model.
            results (Dict[str, Any]): The results of training and validation.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_evaluation_complete(
        self,
        model: "AbstractRecommender",
        results: Dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ):
        """Callback method to be called after evaluation is complete (for a specific test fold).

        Args:
            model (AbstractRecommender): The model that has been evaluated.
            results (dict): The results of the evaluation.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """

    def on_model_complete(
        self,
        model_name: str,
        test_results: List[Dict[str, Any]],
        trials: List[List[Any]],
        *args: Any,
        **kwargs: Any
    ):
        """Callback method to be called after training and testing are complete (for a specific model).

        Args:
            model_name (str): The name of the model.
            test_results (List[Dict[str, Any]]): The results of testing, for each test fold.
            trials (List[List[Any]]): The trials, for each test fold.
            *args (Any): Additional positional arguments.
            **kwargs (Any): Additional keyword arguments.
        """


class CallbackManager:
    """Class to manage and trigger a collection of callbacks for specific events.

    This class handles the storage of callback objects and provides a mechanism to
    invoke their methods dynamically based on event names.

    Attributes:
        callbacks (List[ElliotCallback]): List of callback objects, already sorted by priority.
    """

    def __init__(self, callbacks: List[ElliotCallback]):
        self.callbacks = callbacks

    def trigger(
        self,
        event_name: str,
        *args: Any,
        **kwargs: Any
    ):
        """Trigger callbacks for a specific event.

        Args:
            event_name (str): The name of the event to trigger callbacks for.
            *args (Any): Additional positional arguments to be passed to the callbacks.
            **kwargs (Any): Additional keyword arguments to be passed to the callbacks.
        """
        for cb in self.callbacks:
            method = getattr(cb, event_name, None)
            if callable(method):
                method(*args, **kwargs)
