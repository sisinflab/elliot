from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING
import pandas as pd
import logging as pylog

from elliot.dataset.modular_loaders import SideInformation
from elliot.dataset.interactions import Interactions
from elliot.dataset.sessions import Sessions
from elliot.namespace import ExperimentConfig
from elliot.utils import logging

if TYPE_CHECKING:
    from elliot.evaluation.protocol import EvalProtocol


class DataSet:
    """Train/eval pair for one fold. Wraps the fold's `Interactions` (train and eval)
    and `Sessions` (train) views, sharing the fold's (user, item) mappings. How the
    eval view is evaluated is up to the fold's `EvalProtocol`, attached from outside
    as `protocol` by `elliot.evaluation.protocol.prepare_protocols()`: the dataset
    only hosts it, so that it lives (and is shared) exactly as long as the fold.

    Args:
        config (ExperimentConfig): Experiment configuration object.
        train_data (Union[pd.DataFrame, Interactions]): Train interactions, either as a raw
            DataFrame (built fresh) or an already-built `Interactions` (shared, e.g.
            across validation folds of the same test fold).
        eval_data (pd.DataFrame): Eval (test/validation) interactions.
        side_info (SideInformation, optional): Shared, centrally-cached side-information handle.
            Defaults to None.
        evaluation_set (str): Name of this fold's eval split ("test" or "validation").
            Defaults to "test".
        fold_index (Tuple[int, Optional[int]]): Tuple containing the complete fold index.
            Defaults to `(0, None)`.
    """

    train_set: Interactions
    eval_set: Interactions
    train_sessions: Sessions

    def __init__(
        self,
        config: ExperimentConfig,
        train_data: pd.DataFrame,
        eval_data: pd.DataFrame,
        side_info: Optional[SideInformation] = None,
        evaluation_set: str = "test",
        fold_index: Tuple[int, Optional[int]] = (0, None),
        *args,
        **kwargs
    ):
        self.logger = logging.get_logger(
            self.__class__.__name__, pylog.CRITICAL if config.config_test else pylog.DEBUG
        )

        # Initializing variables
        self.config = config
        self.args = args
        self.kwargs = kwargs

        self._users: List[Any] = []
        self._items: List[Any] = []
        self._p_users: List[Any] = []
        self._p_items: List[Any]  = []
        self._u_map: Dict[Any, int] = {}
        self._i_map: Dict[Any, int] = {}
        self._evaluation_set = evaluation_set
        self._protocol: Optional["EvalProtocol"] = None

        self.fold_index = fold_index

        # Discover this fold's (user, item) domain from the raw train DataFrame
        self._users = (
            train_data["userId"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        self._items = (
            train_data["itemId"]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )

        # Assign each public user/item id a private, 0-based index
        self._p_users = list(range(len(self._users)))
        self._p_items = list(range(len(self._items)))

        self._u_map = {user: k for k, user in zip(self._p_users, self._users)}
        self._i_map = {item: k for k, item in zip(self._p_items, self._items)}

        # Build the train view from raw data
        self.train_set = Interactions(
            dataframe=train_data,
            name="train",
            mappings=self.get_mappings(),
            inv_mappings=self.get_inverse_mappings(),
            side_info=side_info
        )

        num_users, num_items = self.train_set.dims
        transactions = self.train_set.transactions
        sparsity = 1 - (transactions / (num_users * num_items))

        self.logger.info(
            f"Statistics\t"
            f"Users:\t{num_users}\t"
            f"Items:\t{num_items}\t"
            f"Transactions:\t{transactions}\t"
            f"Sparsity:\t{sparsity}"
        )

        # Log each loader's user/item coverage against this fold's train domain
        if side_info is not None:
            for name, loader in side_info.items():
                mapped_users, mapped_items = loader.get_mapped()
                missing_users = len(set(self.train_set.get_dict().keys()) - set(mapped_users))
                missing_items = len(set(self._i_map.keys()) - set(mapped_items))
                self.logger.info(
                    "Side information aligned",
                    extra={
                        "context": {
                            "source": name,
                            "users_in_side": len(mapped_users),
                            "items_in_side": len(mapped_items),
                            "missing_users_vs_train": missing_users,
                            "missing_items_vs_train": missing_items,
                            "alignment_mode": getattr(loader, "alignment", None),
                            "materialization": getattr(loader, "materialization", None),
                        }
                    },
                )

        # Build the eval view, sharing this fold's (user, item) mappings with train
        self.eval_set = Interactions(
            dataframe=eval_data,
            name=self._evaluation_set,
            mappings=self.get_mappings(),
            inv_mappings=self.get_inverse_mappings(),
            side_info=side_info
        )

        # Build the ordered, per-user/session view of train for sequential models
        self.train_sessions = Sessions(
            dataframe=self.train_set.dataframe,
            name="train",
            mappings=self.get_mappings(),
            sparse=self.train_set.sparse_ratings
        )

    @property
    def evaluation_set(self) -> str:
        """Name of this fold's eval split ("test" or "validation")."""
        return self._evaluation_set

    @property
    def protocol(self) -> "EvalProtocol":
        """This fold's evaluation protocol.

        Raises:
            RuntimeError: If no protocol has been attached to this dataset yet.
        """
        if self._protocol is None:
            raise RuntimeError("This DataSet has no evaluation protocol: call "
                               "`elliot.evaluation.protocol.prepare_protocols()` first.")
        return self._protocol

    @protocol.setter
    def protocol(self, protocol: "EvalProtocol"):
        self._protocol = protocol

    def get_mappings(self) -> Tuple[Dict[Any, int], Dict[Any, int]]:
        """Return this fold's (user, item) public-to-private id mappings.

        Returns:
            Tuple[Dict[Any, int], Dict[Any, int]]: The (user, item) mappings.
        """
        return self._u_map, self._i_map

    def get_inverse_mappings(self) -> Tuple[List[Any], List[Any]]:
        """Return this fold's (user, item) private-to-public id mappings, as lists
        indexed by private id.

        Returns:
            Tuple[List[Any], List[Any]]: The (user, item) inverse mappings.
        """
        return self._users, self._items

    def get_users_items(self, private: bool = False) -> Tuple[List[Any], List[Any]]:
        """Return this fold's (user, item) domain.

        Args:
            private (bool): If True, return private (index-space) ids; otherwise
                public ids. Defaults to False.

        Returns:
            Tuple[List[Any], List[Any]]: The (users, items) in this fold's domain.
        """
        return (self._users, self._items) if not private else (self._p_users, self._p_items)
