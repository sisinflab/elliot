from typing import TypeVar, Dict, Type, Optional, Callable, List, Generic, TYPE_CHECKING, Tuple, Any

if TYPE_CHECKING:
    from elliot.dataset.modular_loaders import AbstractLoader
    from elliot.dataset.samplers import AbstractSampler
    from elliot.evaluation.metrics import BaseMetric
    from elliot.recommender import AbstractRecommender
    from elliot.utils.callback import ElliotCallback


# BASIC REGISTRY

T = TypeVar("T")


class BasicRegistry(Generic[T]):
    """Basic registry with functionality to store information.

    Args:
        registry_name (str): Name of the registry.
    """

    def __init__(self, registry_name: str):
        self._registry: Dict[str, Type[T]] = {}
        self.registry_name = registry_name

    def register(self, name: Optional[str] = None, **kwargs: Any) -> Callable:
        """Decorator to register a class in the registry.

        Args:
            name (str, optional): Name for registration. If None, use class name.
            **kwargs (Any): Additional keyword arguments to be attached to the class.

        Returns:
            Callable: The decorator to register new data.
        """

        def decorator(cls: Type[T]) -> Type[T]:
            """The definition of the decorator.

            Args:
                cls (Type[T]): Any type of class to be stored.

            Returns:
                Type[T]: Any type of class.
            """
            nonlocal name
            key = name or cls.__name__
            for k, v in kwargs.items():
                setattr(cls, k, v)
            self._registry[key] = cls
            return cls

        return decorator

    def get(self, name: str, *args: Any, **kwargs: Any) -> T:
        """Get an instance from the registry by name.

        Args:
            name (str): Name of the registered class.
            *args (Any): Arguments to pass to the class constructor.
            **kwargs (Any): Keyword arguments to pass to the class constructor.

        Returns:
            T: Any type of object stored previously.

        Raises:
            ValueError: If name is not to be found in registry.
        """
        cls = self._registry.get(name)
        if cls is None:
            raise ValueError(
                f"'{name}' not found in {self.registry_name} registry. "
                f"Available options: {list(self._registry.keys())}"
            )
        return cls(*args, **kwargs)

    def get_class(self, name: str) -> Type[T]:
        """Get the class from the registry by name.

        Args:
            name (str): Name of the registered class.

        Returns:
            Type[T]: Any type of object previously stored.

        Raises:
            ValueError: If name is not to be found in registry.
        """
        cls = self._registry.get(name)
        if cls is None:
            raise ValueError(
                f"'{name}' not found in {self.registry_name} registry. "
                f"Available options: {list(self._registry.keys())}"
            )
        return cls

    def all(self) -> List[str]:
        """List all registered names.

        Returns:
            List[str]: The list of names stored.
        """
        return list(self._registry.keys())


# Singleton basic registries
model_registry: BasicRegistry["AbstractRecommender"] = BasicRegistry("Recommender")
side_info_registry: BasicRegistry["AbstractLoader"] = BasicRegistry("SideInfo")
sampler_registry: BasicRegistry["AbstractSampler"] = BasicRegistry("Sampler")
metric_registry: BasicRegistry["BaseMetric"] = BasicRegistry("Metric")


# CALLBACK REGISTRY

CallbackT = TypeVar("CallbackT", bound="ElliotCallback")


class CallbackRegistry(Generic[CallbackT]):
    """Callback registry with functionality to store information.

    This class provides mechanisms to register callback classes with associated metadata and a
    priority value. Additionally, it allows listing and instantiating these registered callbacks
    in priority order.

    Args:
        registry_name (str): Name of the registry.
    """

    def __init__(self, registry_name: str):
        self._registry: Dict[str, Tuple[Type[CallbackT], int]] = {}
        self.registry_name = registry_name

    def register(
        self,
        name: Optional[str] = None,
        priority: Optional[int] = None,
        **kwargs: Any
    ) -> Callable:
        """Decorator to register an ElliotCallback class in the registry with metadata and priority.

        Args:
            name (str, optional): Name for registration. If None, use class name.
            priority (int, optional): Priority for ordering. If None, use the next available priority.
            **kwargs (Any): Keyword arguments to be attached to the class.

        Returns:
            Callable: The decorator to register new data.
        """

        def decorator(cls: Type[CallbackT]) -> Type[CallbackT]:
            """The definition of the decorator.

            Args:
                cls (Type[CallbackT]): ElliotCallback class to register.

            Returns:
                Type[CallbackT]: The ElliotCallback class.

            Raises:
                ValueError: If the class name or specified name is already registered.
            """
            nonlocal name
            key = name or cls.__name__

            if key in self._registry:
                raise ValueError(f"{key} already registered in {self.registry_name}")

            # Attach metadata to class
            for k, v in kwargs.items():
                setattr(cls, k, v)

            # Define priority
            nonlocal priority
            current_max_priority = (
                max(v[1] for v in self._registry.values())
                if self._registry else 0
            )
            priority = priority or (current_max_priority + 1)

            self._registry[key] = (cls, priority)
            return cls

        return decorator

    def get_all(self, *args: Any, **kwargs: Any) -> List[CallbackT]:
        """Instantiate all callbacks ordered by priority (high -> low).

        Args:
            *args (Any): Arguments to pass to the class constructor.
            **kwargs (Any): Keyword arguments to pass to the class constructor.
        """
        sorted_values = sorted(
            self._registry.values(),
            key=lambda value: value[1]
        )

        return [cls(*args, **kwargs) for cls, _ in sorted_values]

    def all(self) -> List[str]:
        """List all registered names.

        Returns:
            List[str]: The list of names stored.
        """
        return list(self._registry.keys())


# Singleton callback registries
callback_registry: CallbackRegistry = CallbackRegistry("Callback")
