import torch
from torch import nn
from torch.nn import functional as F

from elliot.dataset import Interactions
from elliot.dataset.modular_loaders.materialize import dedupe_edges
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import GraphBasedRecommender, KnowledgeAwareRecommender
from elliot.recommender.init import xavier_uniform_init
from elliot.recommender.layers import EdgeDropout, RelationWeightedMeanAggregator, SparseDropout
from elliot.utils.registry import model_registry


class Aggregator(nn.Module):
    """Relational path-aware convolution: aggregates entities through the
    knowledge graph (relation-weighted mean over each entity's neighbours),
    and aggregates users through the interaction graph, disentangled over
    `n_factors` latent user-intent channels.
    """

    def __init__(self, n_users: int, n_factors: int):
        super().__init__()
        self.n_users = n_users
        self.n_factors = n_factors
        self.mean_aggregator = RelationWeightedMeanAggregator()

    def forward(
        self,
        entity_emb,
        user_emb,
        latent_emb,
        edge_index,
        edge_type,
        interact_mat,
        weight,
        disen_weight_att
    ):
        channel = entity_emb.shape[1]

        # KG aggregate: relation-weighted mean of each entity's incoming neighbors
        entity_agg = self.mean_aggregator(entity_emb, edge_index, edge_type, weight)

        # user->latent factor attention
        score_ = user_emb @ latent_emb.T
        score = F.softmax(score_, dim=1).unsqueeze(-1)  # [n_users, n_factors, 1]

        # user aggregate
        user_agg = torch.sparse.mm(interact_mat, entity_emb)  # [n_users, channel]
        disen_weight = (F.softmax(disen_weight_att, dim=1) @ weight).unsqueeze(0).expand(
            self.n_users, self.n_factors, channel
        )
        user_agg = user_agg * (disen_weight * score).sum(dim=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg


class GraphConv(nn.Module):
    """Graph convolutional network: stacks `n_hops` `Aggregator` layers,
    accumulating a residual sum of every layer's (L2-normalized) output.
    """

    def __init__(
        self,
        channel,
        n_hops,
        n_users,
        n_factors,
        n_relations,
        ind,
        node_dropout_rate: float = 0.5,
        mess_dropout_rate: float = 0.1
    ):
        super().__init__()

        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        # relation-id 0 ("interacts") excluded: it has no learned embedding
        self.weight = nn.Parameter(torch.empty(n_relations - 1, channel))
        self.disen_weight_att = nn.Parameter(torch.empty(n_factors, n_relations - 1))

        self.convs = nn.ModuleList(
            [Aggregator(n_users=n_users, n_factors=n_factors) for _ in range(n_hops)]
        )
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.edge_dropout = EdgeDropout(node_dropout_rate)
        self.sparse_dropout = SparseDropout(node_dropout_rate)

    def _cul_cor(self):
        def cosine_similarity(tensor_1, tensor_2):
            normalized_tensor_1 = tensor_1 / torch.norm(tensor_1, dim=0, keepdim=True)
            normalized_tensor_2 = tensor_2 / torch.norm(tensor_2, dim=0, keepdim=True)
            return (normalized_tensor_1 * normalized_tensor_2).sum(dim=0) ** 2  # no negative

        def distance_correlation(tensor_1, tensor_2):
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            tensor_1, tensor_2 = tensor_1.unsqueeze(-1), tensor_2.unsqueeze(-1)

            a_, b_ = (tensor_1 @ tensor_1.T) * 2, (tensor_2 @ tensor_2.T) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a = torch.sqrt(torch.clamp(tensor_1_square - a_ + tensor_1_square.T, min=0.0) + 1e-8)
            b = torch.sqrt(torch.clamp(tensor_2_square - b_ + tensor_2_square.T, min=0.0) + 1e-8)

            A = a - a.mean(dim=0, keepdim=True) - a.mean(dim=1, keepdim=True) + a.mean()
            B = b - b.mean(dim=0, keepdim=True) - b.mean(dim=1, keepdim=True) + b.mean()
            dcov_AB = torch.sqrt(torch.clamp((A * B).sum() / channel ** 2, min=0.0) + 1e-8)
            dcov_AA = torch.sqrt(torch.clamp((A * A).sum() / channel ** 2, min=0.0) + 1e-8)
            dcov_BB = torch.sqrt(torch.clamp((B * B).sum() / channel ** 2, min=0.0) + 1e-8)
            return (dcov_AB / torch.sqrt(dcov_AA * dcov_BB + 1e-8)).squeeze()

        def mutual_information():
            disen_T = self.disen_weight_att.T  # [n_factors, dimension]
            normalized_disen_T = F.normalize(disen_T, p=2, dim=1)

            # Self-similarity of a unit vector is always 1 (a constant, not a
            # measured quantity); normalizing ttl_scores too keeps every entry
            # in [-1, 1] regardless of how large disen_weight_att grows.
            pos_scores = (normalized_disen_T * normalized_disen_T).sum(dim=1)
            ttl_scores = (normalized_disen_T @ normalized_disen_T.T).sum(dim=1)

            # log(exp(a) / exp(b)) == a - b: skip the exp/log round trip so
            # there is nothing left to overflow.
            return -torch.sum(pos_scores - ttl_scores) / self.temperature

        # similarity for each latent factor weight pairs
        if self.ind == 'mi':
            return mutual_information()

        cor = 0.
        for i in range(self.n_factors):
            for j in range(i + 1, self.n_factors):
                if self.ind == 'distance':
                    cor = cor + distance_correlation(self.disen_weight_att[i], self.disen_weight_att[j])
                else:
                    cor = cor + cosine_similarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    def forward(
        self,
        user_emb,
        entity_emb,
        latent_emb,
        edge_index,
        edge_type,
        interact_mat,
        mess_dropout=True,
        node_dropout=False
    ):
        # node dropout
        if node_dropout:
            edge_index, edge_type = self.edge_dropout(edge_index, edge_type)
            interact_mat = self.sparse_dropout(interact_mat)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()

        for conv in self.convs:
            entity_emb, user_emb = conv(
                entity_emb, user_emb, latent_emb, edge_index, edge_type,
                interact_mat, self.weight, self.disen_weight_att
            )

            # message dropout
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb, p=2, dim=1)
            user_emb = F.normalize(user_emb, p=2, dim=1)

            # result emb
            entity_res_emb = entity_res_emb + entity_emb
            user_res_emb = user_res_emb + user_emb

        return entity_res_emb, user_res_emb, cor


@model_registry.register()
class KGIN(KnowledgeAwareRecommender, GraphBasedRecommender):
    """
    Learning Intents behind Interactions with Knowledge Graph for Recommendation

    For further details, please refer to the `paper <https://arxiv.org/abs/2102.07057>`_

    Args:
        learning_rate: Learning rate
        epochs: Number of epochs
        factors: Embedding size
        batch_size: Batch size
        decay: L2 regularization weight for the user/entity embeddings
        sim_decay: Regularization weight for the latent-factor independence loss
        n_layers: Number of context hops (graph convolution layers)
        n_factors: Number of latent factors for user intent
        node_dropout: Whether to apply node dropout
        node_dropout_rate: Node dropout ratio
        mess_dropout: Whether to apply message dropout
        mess_dropout_rate: Message dropout ratio
        ind: Independence modeling: mi, distance, cosine

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        KGIN:
          meta:
            save_recs: True
          learning_rate: 0.0001
          epochs: 50
          batch_size: 1024
          factors: 64
          n_layers: 3
          n_factors: 4
          decay: 0.00001
          sim_decay: 0.0001
          node_dropout: True
          node_dropout_rate: 0.5
          mess_dropout: True
          mess_dropout_rate: 0.1
          ind: distance
    """

    loaders = ["KGTriplesLoader"]

    # Model hyperparameters
    factors: int = 64
    n_layers: int = 3
    n_factors: int = 4
    learning_rate: float = 1e-4
    decay: float = 1e-5
    sim_decay: float = 1e-4
    node_dropout: bool = True
    node_dropout_rate: float = 0.5
    mess_dropout: bool = True
    mess_dropout_rate: float = 0.1
    ind: str = "distance"

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super(KGIN, self).__init__(params, seed, interactions, *args, **kwargs)

        # Relation-id 0 is reserved here for the "interacts" relation,
        # so GraphConv can size its per-relation weight to exclude it
        self.n_relations = self.n_relations + 1

        # KG relation graph: the (head, tail) edges plus their parallel relation type,
        # de-duplicated since relation-aware propagation expects one edge per distinct triple,
        # not per occurrence in the raw data
        edge_index = torch.stack([self.kg_heads, self.kg_tails], dim=0)
        self.edge_index, self.edge_type = dedupe_edges(edge_index, self.kg_relations, device=self._device)

        # Shift relation ids by 1 to keep id 0 reserved for "interacts"
        self.edge_type = self.edge_type + 1

        # Row-normalized (D^{-1}A) user -> entity interaction matrix
        self.interact_mat = self.build_interact_mat()

        # Embeddings
        self.user_embed = nn.Embedding(self._num_users, self.factors)
        self.entity_embed = nn.Embedding(self.n_entities, self.factors)
        self.latent_emb = nn.Embedding(self.n_factors, self.factors)

        self.gcn = GraphConv(
            channel=self.factors,
            n_hops=self.n_layers,
            n_users=self._num_users,
            n_factors=self.n_factors,
            n_relations=self.n_relations,
            ind=self.ind,
            node_dropout_rate=self.node_dropout_rate,
            mess_dropout_rate=self.mess_dropout_rate
        )

        # Loss and optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Sampler configuration
        self.sampler_config = {
            "name": "PairWiseSampler"
        }

        # Init embedding weights
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def forward(self):
        entity_gcn_emb, user_gcn_emb, cor = self.gcn(
            self.user_embed.weight,
            self.entity_embed.weight,
            self.latent_emb.weight,
            self.edge_index,
            self.edge_type,
            self.interact_mat,
            self.training and self.mess_dropout,
            self.training and self.node_dropout
        )
        return user_gcn_emb, entity_gcn_emb, cor

    def train_step(self, batch, *args):
        user, pos, neg = [x.to(self._device) for x in batch]
        batch_size = user.shape[0]

        user_gcn_emb, entity_gcn_emb, cor = self.forward()

        u_e = user_gcn_emb[user]
        pos_e = entity_gcn_emb[self.item_entity_ids[pos]]
        neg_e = entity_gcn_emb[self.item_entity_ids[neg]]

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)

        difference = torch.clamp(pos_scores - neg_scores, -80.0, 1e8)
        mf_loss = F.softplus(-difference).mean()

        reg_loss = self.decay * (
            u_e.pow(2).sum() + pos_e.pow(2).sum() + neg_e.pow(2).sum()
        ) / 2 / batch_size
        cor_loss = self.sim_decay * cor

        loss = mf_loss + reg_loss + cor_loss

        return loss

    def predict(self, user_indices, item_indices=None, **kwargs):
        user_gcn_emb, entity_gcn_emb = self.propagate_embeddings()[:2]

        user_embeddings = user_gcn_emb[user_indices]

        if item_indices is None:
            item_embeddings = entity_gcn_emb[self.item_entity_ids]
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = entity_gcn_emb[self.item_entity_ids[item_indices.clamp(min=0)]]
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(einsum_string, user_embeddings, item_embeddings)
        return predictions
