from typing import List, Tuple, Optional, no_type_check, Union, Any
import inspect
import random
import logging as pylog
import numpy as np
import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from abc import ABC, abstractmethod

from elliot.dataset import Interactions, Sessions
from elliot.dataset.modular_loaders.materialize import graph_item_entity_ids_to_tensor
from elliot.dataset.samplers import CombinedDataLoader
from elliot.namespace import RecommenderConfig
from elliot.recommender.init import zeros_init
from elliot.recommender.layers import SparseAdjacency
from elliot.utils import get_device, logging
from elliot.utils.enums import ModelType, SamplerType
from elliot.utils.registry import sampler_registry
from elliot.utils.read import Reader
from elliot.utils.write import Writer


class AbstractRecommender(ABC):
    type: ModelType
    sampler_config: dict = {}
    side_info_sampler_config: dict = {}
    loaders: List[str] = []

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        self._interactions = interactions
        self._seed = seed
        self._users, self._items = interactions.get_users_items()
        self._num_users, self._num_items = interactions.dims
        self.transactions = interactions.transactions

        package_name = inspect.getmodule(self.__class__).__package__
        cls_name = self.__class__.__name__
        rec_name = f"external.{cls_name}" if "external" in package_name else cls_name
        self.logger = logging.get_logger_model(rec_name, pylog.DEBUG)

        self.reader = Reader(self.logger)
        self.writer = Writer(self.logger)

        self.model_config = params
        self.params_list = []

        self.set_seed(seed)
        self.set_params(params)

        missing = [name for name in self.loaders if name not in self._interactions.side_information]
        if missing:
            raise KeyError(
                f"{self.__class__.__name__} declares `loaders` entries not present in "
                f"the dataset's side information: {missing}."
            )

        self.model_config.name = self.name

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def name_param(self):
        """The name of the model with all it's parameters."""
        name = ""
        for ann, _ in self.__class__.__annotations__.items():
            value = getattr(self, ann, None)
            if isinstance(value, float):
                name += f"_{ann}={value:.4f}"
            else:
                name += f"_{ann}={value}"
        return name

    @abstractmethod
    def set_seed(self, seed: int):
        raise NotImplementedError()

    def set_params(self, params: RecommenderConfig):
        self.logger.info("Loading parameters")

        for name, val in params.model_dump().items():
            if name in self.__class__.__annotations__:
                setattr(self, name, val)
                self.logger.info(f"Parameter '{name}' set to {val}")
                self.params_list.append(name)

    @abstractmethod
    def get_model_state(self):
        raise NotImplementedError()

    @abstractmethod
    def set_model_state(self, checkpoint):
        raise NotImplementedError()

    @abstractmethod
    def get_training_dataloader(self, batch_size):
        raise NotImplementedError()

    @abstractmethod
    def train_step(self, batch, *args):
        raise NotImplementedError()

    @no_type_check
    @abstractmethod
    def predict(self, *args, item_indices=None, **kwargs):
        raise NotImplementedError()

    def _check_sampler(self, allowed_types, config: Optional[dict] = None):
        config = self.sampler_config if config is None else config
        try:
            sampler_name = config.pop("name")
        except KeyError:
            raise ValueError(
                f"Sampler name is not specified for {self.__class__.__name__}. "
                f"Please provide a valid 'name' field in the sampler configuration."
            )
        sampler_class = sampler_registry.get_class(sampler_name)
        if not isinstance(allowed_types, tuple):
            allowed_types = (allowed_types,)
        if sampler_class.type not in allowed_types:
            raise ValueError(
                f"Sampler '{sampler_name}' is not compatible with {self.__class__.__name__}. "
                f"Please use a sampler of type {'or '.join([t.name for t in allowed_types])}."
            )
        return sampler_name

    def _check_for_side_info_dataloader(
        self,
        dataloader: DataLoader[Any],
        batch_size: int
    ) -> Union[DataLoader[Any], CombinedDataLoader]:
        """Wrap `dataloader` together with a second one built from
        `self.side_info_sampler_config`, when the model declares one.

        This is how side information (e.g. a knowledge graph) gets its own,
        entirely independent sampler instead of being folded into the
        interaction sampler producing `dataloader`: the two are sampled, batched
        and cached separately, and only zipped together (via
        `CombinedDataLoader`) into the combined batch a model's `train_step` sees.

        Args:
            dataloader (DataLoader[Any]): The (already built) primary training dataloader.
            batch_size (int): Batch size for the side-information dataloader.

        Returns:
            Union[DataLoader[Any], CombinedDataLoader]: The unwrapped `dataloader`,
                or a `CombinedDataLoader` pairing it with the side-information dataloader.
        """
        if not self.side_info_sampler_config:
            return dataloader

        side_info_sampler_config = dict(self.side_info_sampler_config)
        sampler_name = self._check_sampler(
            allowed_types=(SamplerType.TRADITIONAL, SamplerType.PIPELINE),
            config=side_info_sampler_config
        )
        side_info_dataloader = self._interactions.get_side_info_dataloader(
            sampler_name=sampler_name,
            batch_size=batch_size,
            seed=self._seed,
            **side_info_sampler_config
        )
        return CombinedDataLoader(dataloader, side_info_dataloader)


class BaseRecommender(AbstractRecommender):
    type = ModelType.BASE

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)
        self.modules = []
        self.bias = []
        self.params_to_save = []

    def set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)

    def apply(self, init_func, **kwargs):
        for m in self.modules:
            if any(m is x for x in self.bias):
                zeros_init(m)
            else:
                init_func(m, **kwargs)

    def get_model_state(self):
        return {p: getattr(self, p) for p in self.params_to_save}

    def set_model_state(self, checkpoint):
        for k, v in checkpoint:
            if k in self.params_to_save:
                setattr(self, k, v)

    def get_training_dataloader(self, batch_size):
        if self.sampler_config:
            sampler_name = self._check_sampler(
                allowed_types=(SamplerType.TRADITIONAL, SamplerType.PIPELINE)
            )
            dataloader = self._interactions.get_dataloader(
                sampler_name=sampler_name,
                batch_size=batch_size,
                seed=self._seed,
                **self.sampler_config
            )
            return self._check_for_side_info_dataloader(dataloader, batch_size)
        else:
            for _ in range(1):
                yield None

    @abstractmethod
    def predict(self, user_indices, item_indices=None, **kwargs):
        raise NotImplementedError()


class TraditionalRecommender(BaseRecommender):
    type = ModelType.TRADITIONAL

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)
        self.similarity_matrix = None
        self._train = self._interactions.sparse_ratings
        self._implicit_train = self._interactions.sparse

    def get_training_dataloader(self, batch_size):
        for _ in range(1):
            yield None

    def train_step(self, *args):
        pass

    @abstractmethod
    def initialize(self):
        raise NotImplementedError()


class GeneralRecommender(nn.Module, AbstractRecommender):
    type = ModelType.GENERAL

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        AbstractRecommender.__init__(self, params, seed, interactions, *args, **kwargs)
        super(GeneralRecommender, self).__init__()
        self.bias = []
        self._device = get_device()

    def set_seed(self, seed: int):
        """Set random seed for reproducibility.

        Args:
            seed (int): The seed value to be used.
        """
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def apply(self, init_func, **kwargs):
        for m in self.modules():
            if any(m is x for x in self.bias):
                zeros_init(m)
            else:
                init_func(m, **kwargs)

    def get_model_state(self):
        return {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }

    def set_model_state(self, checkpoint):
        self.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def get_training_dataloader(self, batch_size):
        sampler_name = self._check_sampler(
            allowed_types=(SamplerType.TRADITIONAL, SamplerType.PIPELINE)
        )
        dataloader = self._interactions.get_dataloader(
            sampler_name=sampler_name,
            batch_size=batch_size,
            seed=self._seed,
            **self.sampler_config
        )
        return self._check_for_side_info_dataloader(dataloader, batch_size)

    @abstractmethod
    def predict(self, user_indices, item_indices=None, **kwargs):
        raise NotImplementedError()


class GraphBasedRecommender(GeneralRecommender):
    # Cache storage
    _cached_user_emb: Optional[Tensor] = None
    _cached_item_emb: Optional[Tensor] = None

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)

    def train(self, mode=True):
        """Override train mode to empty the cache when switching to training."""
        super().train(mode)

        if mode:
            # We are in training mode, embeddings will change. Empty the cache
            self._cached_user_emb = None
            self._cached_item_emb = None

    def propagate_embeddings(self) -> Tuple[Tensor, Tensor]:
        """Retrieve the propagate user and item embeddings.

        Subsequent calls will return the cached values, speeding up the
        evaluation process.

        Returns:
            Tuple[Tensor, Tensor]: (User Embeddings, Item Embeddings)
        """
        # Safety check
        if self.training:
            return self.forward()

        # Check if values are cached
        if self._cached_user_emb is None or self._cached_item_emb is None:
            with torch.no_grad():
                # Unpack the forward
                ret = self.forward()
                self._cached_user_emb = ret[0]
                self._cached_item_emb = ret[1]

        return self._cached_user_emb, self._cached_item_emb

    def get_adj_mat(self, normalize: bool = False) -> SparseAdjacency:
        """Get the interaction matrix of users and items as a sparse adjacency.

        Args:
            normalize (bool): Whether to apply symmetric normalization
                (D^-0.5 * A * D^-0.5) to the adjacency.

        Returns:
            SparseAdjacency: The sparse adjacency matrix.
        """
        # Extract user and items nodes
        row, col = self._interactions.sparse.nonzero()
        user_nodes = row
        item_nodes = col + self._num_users

        # Unify arcs in both directions
        row = np.concatenate([user_nodes, item_nodes])
        col = np.concatenate([item_nodes, user_nodes])

        # Create the edge tensor
        edge_index_np = np.vstack([row, col])
        # Creating a tensor directly from a numpy array instead of lists
        edge_index = torch.tensor(edge_index_np, dtype=torch.int64)

        size = self._num_items + self._num_users

        adj = SparseAdjacency(
            row=edge_index[0],
            col=edge_index[1],
            size=(size, size),
        ).to(self._device)

        if normalize:
            adj = self._symmetric_normalization(adj)

        return adj

    @staticmethod
    def _symmetric_normalization(adj: SparseAdjacency) -> SparseAdjacency:
        """Applies symmetric normalization: D^-0.5 * A * D^-0.5.

        Args:
            adj (SparseAdjacency): The adjacency to normalize.

        Returns:
            SparseAdjacency: The normalized adjacency.
        """
        deg = adj.sum(dim=1)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0.0)

        row, col, _ = adj.coo()
        norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return adj.set_value(norm_vals)

    def get_ego_embeddings(
        self, user_embedding: nn.Embedding, item_embedding: nn.Embedding
    ) -> Tensor:
        """Get the initial embedding of users and items and combine to an embedding matrix.

        Args:
            user_embedding (nn.Embedding): The user embeddings.
            item_embedding (nn.Embedding): The item embeddings.

        Returns:
            Tensor: Combined user and item embeddings.
        """
        user_embeddings = user_embedding.weight
        item_embeddings = item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


class KnowledgeAwareRecommender(GeneralRecommender):
    """Base class for recommenders consuming `KGTriplesLoader`'s `kg_triples`
    payload: exposes the KG's entity/relation counts, the item -> KG entity bridge
    (`item_entity_ids`), and the raw `(head, relation, tail)` triples as parallel
    `int64` tensors - the common ground any KG-aware model builds on, regardless of
    how it actually consumes the KG (translational embeddings, graph propagation, ...).
    """

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)

        payload = self._interactions.get_side_info(self.loaders[0])["kg_triples"]

        self.n_entities = payload.n_entities
        self.n_relations = payload.n_relations

        self.item_entity_ids = graph_item_entity_ids_to_tensor(payload, device=self._device)

        self.kg_heads = torch.as_tensor(payload.heads, dtype=torch.long, device=self._device)
        self.kg_relations = torch.as_tensor(payload.relations, dtype=torch.long, device=self._device)
        self.kg_tails = torch.as_tensor(payload.tails, dtype=torch.long, device=self._device)

    def build_interact_mat(self) -> torch.Tensor:
        """Build the row-normalized (D^{-1}A) user -> entity interaction matrix,
        i.e. the (n_users x n_entities) block of the full user/entity adjacency,
        remapping `self._interactions.sparse_tensor`'s item columns to their KG
        entity ids (row-normalization only ever mixes a user row with its own,
        entity-only, non-zero columns, so it is equivalent to normalize this block
        directly).

        Returns:
            torch.Tensor: The sparse (n_users x n_entities) interaction matrix.
        """
        row, col = self._interactions.sparse_tensor.indices()
        row, col = row.to(self._device), col.to(self._device)
        col = self.item_entity_ids[col]

        # D^{-1}, using the same per-user degree as the un-remapped matrix
        deg = torch.zeros(self._num_users, device=self._device)
        deg.index_add_(0, row, torch.ones_like(row, dtype=torch.float32))
        d_inv = deg.pow(-1.)
        d_inv[torch.isinf(d_inv)] = 0.

        indices = torch.stack([row, col])
        values = d_inv[row]

        return torch.sparse_coo_tensor(
            indices, values, size=(self._num_users, self.n_entities), device=self._device
        ).coalesce()


class SequentialRecommender(GeneralRecommender):
    """Base class for sequential/session-based models."""
    max_seq_len: int

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        sessions: Sessions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, *args, **kwargs)
        self._sessions = sessions
        self._session_strategy = params.meta.session_strategy

    def get_training_dataloader(self, batch_size):
        sampler_name = self._check_sampler(
            allowed_types=SamplerType.SEQUENTIAL
        )
        dataloader = self._sessions.get_dataloader(
            sampler_name=sampler_name,
            batch_size=batch_size,
            seed=self._seed,
            session_strategy=self._session_strategy,
            **self.sampler_config
        )
        return dataloader

    @abstractmethod
    def predict(self, user_seq, seq_len, item_indices=None, *args, **kwargs):
        raise NotImplementedError()
