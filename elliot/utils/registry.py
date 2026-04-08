from typing import TypeVar, Dict, Type, Optional, Callable, List, Generic, TYPE_CHECKING
if TYPE_CHECKING:
    from elliot.dataset import AbstractLoader, AbstractSampler
    from elliot.evaluation.metrics import BaseMetric
    from elliot.recommender import AbstractRecommender

T = TypeVar("T")


class BasicRegistry(Generic[T]):
    """Basic registry with functionality to store information.

    Args:
        registry_name (str): Name of the registry.
    """

    def __init__(self, registry_name: str):
        self._registry: Dict[str, Type[T]] = {}
        self.registry_name = registry_name

    def register(self, name: Optional[str] = None, **kwargs) -> Callable:
        """Decorator to register a class in the registry.

        Args:
            name (str, optional): Name for registration. If None, use class name.

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

    def get(self, name: str, *args, **kwargs) -> T:
        """Get an instance from the registry by name.

        Args:
            name (str): Name of the registered class.
            *args: Arguments to pass to the class constructor.
            **kwargs: Keyword arguments to pass to the class constructor.

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
            Type[T]: Any type of object stored previously.

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
