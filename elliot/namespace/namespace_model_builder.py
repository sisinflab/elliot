"""
Module description:

"""

__version__ = '0.3.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it'

from abc import ABC, abstractmethod, abstractproperty

from elliot.namespace.namespace_model import NameSpaceModel
from elliot.utils.hydra_config import load_config


class Builder(ABC):
    """
    The Builder interface specifies methods for creating the different parts of
    the Product objects.
    """

    @abstractproperty
    def base(self) -> None:
        pass

    @abstractmethod
    def models(self) -> None:
        pass


class NameSpaceBuilder(Builder):

    def __init__(self, config_path, base_folder_path_elliot, base_folder_path_config, config_overrides=None, config_data=None) -> None:
        """
        A fresh builder instance should contain a blank product object, which is
        used in further assembly.
        """
        config = config_data if config_data is not None else load_config(config_path, overrides=config_overrides)
        self._namespace = NameSpaceModel(config, base_folder_path_elliot, base_folder_path_config)

    @property
    def base(self) -> NameSpaceModel:
        namespace = self._namespace
        namespace.fill_base()
        return namespace

    def models(self) -> tuple:
        return self._namespace.fill_model()
