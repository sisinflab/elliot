import math
import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.utils import softmax as pyg_softmax
from torch_scatter import scatter_mean, scatter_sum

from elliot.dataset import Interactions
from elliot.dataset.modular_loaders.materialize import dedupe_edges
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import GraphBasedRecommender, KnowledgeAwareRecommender
from elliot.recommender.init import xavier_uniform_init
from elliot.recommender.layers import RelationAwareEdgeDropout, RelationWeightedMeanAggregator, SparseDropout
from elliot.utils.registry import model_registry


class KGAttentionAggregator(nn.Module):
    """Single-hop attention-weighted relational convolution: aggregates entities
    through the knowledge graph with a per-edge attention score (a scaled
    dot-product between the head's and the relation-modulated tail's query/key
    projections, softmax-normalized over each entity's incoming edges), and
    aggregates users through the (uniform) interaction graph. Also
    exposes the no-grad `rationale_scores` the masked-edge-modeling and
    contrastive-augmentation tasks build on, since it shares the same attention
    computation as a hop of `forward`.
    """

    def __init__(self, n_heads: int, d_k: int):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_k

    def _edge_logits(
        self,
        entity_emb,
        edge_index,
        edge_type,
        relation_emb,
        W_Q
    ):
        head, tail = edge_index[0], edge_index[1]
        query = (entity_emb[head] @ W_Q).view(-1, self.n_heads, self.d_k)
        key = (entity_emb[tail] @ W_Q).view(-1, self.n_heads, self.d_k)
        key = key * relation_emb[edge_type - 1].view(-1, self.n_heads, self.d_k)
        return (query * key).sum(dim=-1) / math.sqrt(self.d_k)  # [n_edges, n_heads]

    def forward(
        self,
        entity_emb,
        edge_index,
        edge_type,
        interact_mat,
        relation_emb,
        W_Q
    ):
        n_entities = entity_emb.shape[0]
        head, tail = edge_index[0], edge_index[1]

        edge_logits = self._edge_logits(
            entity_emb, edge_index, edge_type, relation_emb, W_Q
        )
        edge_attn = pyg_softmax(edge_logits, head, num_nodes=n_entities)
        value = (entity_emb[tail] * relation_emb[edge_type - 1]).view(-1, self.n_heads, self.d_k)
        weighted = (value * edge_attn.unsqueeze(-1)).view(-1, self.n_heads * self.d_k)

        entity_agg = scatter_sum(weighted, head, dim=0, dim_size=n_entities)
        user_agg = torch.sparse.mm(interact_mat, entity_emb)

        return entity_agg, user_agg

    def rationale_scores(
        self,
        entity_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        relation_emb: torch.Tensor,
        W_Q: torch.Tensor
    ) -> torch.Tensor:
        """No-grad, node-degree-renormalized attention score per edge: how much
        each KG edge explains its head entity's representation, comparable in
        scale across entities of different degree. Feeds the masked-edge-modeling
        and rationale-guided contrastive-augmentation tasks.

        Args:
            entity_emb (torch.Tensor): The `(n_entities, channel)` entity embeddings.
            edge_index (torch.Tensor): The `(2, n_edges)` `(head, tail)` edge index.
            edge_type (torch.Tensor): The parallel `(n_edges,)` relation-type tensor.
            relation_emb (torch.Tensor): The `(n_relations - 1, channel)` shared relation embeddings.
            W_Q (torch.Tensor): The `(channel, channel)` shared query/key projection.

        Returns:
            torch.Tensor: The `(n_edges,)` rationale score per edge.
        """
        with torch.no_grad():
            n_entities = entity_emb.shape[0]
            head = edge_index[0]

            edge_logits = self._edge_logits(
                entity_emb, edge_index, edge_type, relation_emb, W_Q
            ).mean(dim=-1)
            edge_attn_score = pyg_softmax(edge_logits, head, num_nodes=n_entities)

            degree = scatter_sum(
                torch.ones_like(head, dtype=torch.float32), head, dim=0, dim_size=n_entities
            )
            edge_attn_score = edge_attn_score * degree[head]

        return edge_attn_score


class KGAttentionGraphConv(nn.Module):
    """Graph convolutional network: stacks `n_hops` `KGAttentionAggregator` layers,
    accumulating a residual sum of every layer's (L2-normalized) output. Also
    exposes the two single-view propagation modes (`forward_kg`, `forward_ui`) and
    the rationale-score computation (`rationale_scores`) the contrastive and
    masked-edge-modeling tasks build on.
    """

    def __init__(
        self,
        channel: int,
        n_hops: int,
        n_relations: int,
        mess_dropout_rate: float = 0.1
    ):
        super().__init__()

        self.n_hops = n_hops
        self.n_heads = 2
        self.d_k = channel // self.n_heads

        # relation-id 0 ("interacts") excluded: it has no learned embedding
        self.relation_emb = nn.Parameter(torch.empty(n_relations - 1, channel))
        self.W_Q = nn.Parameter(torch.empty(channel, channel))

        self.convs = nn.ModuleList(
            [KGAttentionAggregator(self.n_heads, self.d_k) for _ in range(n_hops)]
        )
        self.dropout = nn.Dropout(p=mess_dropout_rate)
        self.mean_aggregator = RelationWeightedMeanAggregator()

    def rationale_scores(
        self,
        entity_emb: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor
    ) -> torch.Tensor:
        """No-grad rationale score per edge over the shared relation/query
        parameters - see `KGAttentionAggregator.rationale_scores`. Any hop is
        equivalent here since the hops share the same parameters and carry no
        state of their own, so this simply reads it off the first one.
        """
        return self.convs[0].rationale_scores(
            entity_emb, edge_index, edge_type, self.relation_emb, self.W_Q
        )

    def forward(
        self,
        user_emb,
        entity_emb,
        edge_index,
        edge_type,
        interact_mat,
        mess_dropout=True
    ):
        entity_res_emb = entity_emb
        user_res_emb = user_emb

        for conv in self.convs:
            entity_emb, user_emb = conv(
                entity_emb, edge_index, edge_type, interact_mat,
                self.relation_emb, self.W_Q
            )

            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb, p=2, dim=1)
            user_emb = F.normalize(user_emb, p=2, dim=1)

            entity_res_emb = entity_res_emb + entity_emb
            user_res_emb = user_res_emb + user_emb

        return entity_res_emb, user_res_emb

    def forward_kg(
        self,
        entity_emb,
        edge_index,
        edge_type,
        mess_dropout=True
    ):
        """KG-only propagation (plain relation-weighted mean, no attention/users) -
        the contrastive task's KG view of the items.
        """
        entity_res_emb = entity_emb

        for _ in range(self.n_hops):
            entity_emb = self.mean_aggregator(
                entity_emb, edge_index, edge_type, self.relation_emb
            )
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
            entity_emb = F.normalize(entity_emb, p=2, dim=1)
            entity_res_emb = entity_res_emb + entity_emb

        return entity_res_emb

    def forward_ui(
        self,
        user_emb,
        entity_emb,
        interact_mat,
        mess_dropout=True
    ):
        """Interaction-only (bipartite, two-way) propagation, no KG - the
        contrastive task's collaborative-filtering view of the items.
        """
        entity_res_emb = entity_emb
        interact_mat_t = interact_mat.transpose(0, 1).coalesce()

        for _ in range(self.n_hops):
            user_agg = torch.sparse.mm(interact_mat, entity_emb)
            entity_agg = torch.sparse.mm(interact_mat_t, user_emb)
            user_emb, entity_emb = user_agg, entity_agg

            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = F.normalize(entity_emb, p=2, dim=1)
            user_emb = F.normalize(user_emb, p=2, dim=1)

            entity_res_emb = entity_res_emb + entity_emb

        return entity_res_emb


class MaskedEdgeModeling(nn.Module):
    """Graph masked-autoencoding auxiliary task: hides a Gumbel-noised top-k
    rationale-selected subset of KG edges (plus an equally-sized random sample)
    from the encoder graph, then reconstructs them - via a dot product between
    the encoder's contextualized `(head, tail)` entity embeddings and the edge's
    relation embedding - from the encoder's output.
    """

    def __init__(self, mask_size: int):
        super().__init__()
        self.mask_size = mask_size

    def split_edges(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attn_score: torch.Tensor
    ):
        """Rationale-guided (Gumbel top-k) plus random split of the graph into
        an encoder graph and the edges masked out of it for reconstruction.

        Args:
            edge_index (torch.Tensor): The `(2, n_edges)` `(head, tail)` edge index.
            edge_type (torch.Tensor): The parallel `(n_edges,)` relation-type tensor.
            edge_attn_score (torch.Tensor): The `(n_edges,)` rationale score per edge.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: The
            `(encoder_edge_index, encoder_edge_type, masked_edge_index, masked_edge_type)`.
        """
        n_edges = edge_index.shape[1]
        device = edge_index.device

        noise = -torch.log(-torch.log(torch.rand_like(edge_attn_score)))
        msize = min(self.mask_size, n_edges)
        _, topk_edge_id = torch.topk(edge_attn_score + noise, msize, sorted=False)

        topk_mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
        topk_mask[topk_edge_id] = True

        random_id = torch.randperm(n_edges, device=device)[:msize]
        random_mask = torch.zeros(n_edges, dtype=torch.bool, device=device)
        random_mask[random_id] = True

        mask = topk_mask | random_mask
        return edge_index[:, ~mask], edge_type[~mask], edge_index[:, mask], edge_type[mask]

    def forward(
        self, entity_emb: torch.Tensor,
        masked_edge_index: torch.Tensor,
        masked_edge_type: torch.Tensor,
        relation_emb: torch.Tensor
    ) -> torch.Tensor:
        """Reconstruction loss for the edges `split_edges` masked out."""
        head_emb = entity_emb[masked_edge_index[0]]
        tail_emb = entity_emb[masked_edge_index[1]]
        relation = relation_emb[masked_edge_type - 1]
        scores = torch.sigmoid((tail_emb * relation * head_emb).sum(dim=1))
        return -torch.log(scores).mean()


class ContrastHead(nn.Module):
    """InfoNCE-style contrastive head: builds an item's KG-only and
    interaction-only augmented views - dropping the KG edges / interactions the
    model's own attention scores mark as least explanatory - projects each
    through its own small MLP, then pulls the two views together relative to
    every other item in the batch (mismatched pairs).
    """

    def __init__(
        self,
        channel: int,
        tau: float = 1.0,
        drop_ratio: float = 0.5
    ):
        super().__init__()
        self.tau = tau
        self.keep_rate = 1 - drop_ratio
        self.proj_ui = nn.Sequential(
            nn.Linear(channel, channel), nn.ReLU(), nn.Linear(channel, channel)
        )
        self.proj_kg = nn.Sequential(
            nn.Linear(channel, channel), nn.ReLU(), nn.Linear(channel, channel)
        )

    def drop_kg_view(
        self,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_attn_score: torch.Tensor
    ):
        """Rationale-guided KG edge dropout for the KG-only view: drop the
        least-attended edges preferentially, so the augmented view still
        carries the KG's most explanatory structure.
        """
        n_edges = edge_attn_score.shape[0]
        n_drop = n_edges - int(self.keep_rate * n_edges)
        _, drop_id = torch.topk(-edge_attn_score, n_drop, sorted=False)
        mask = torch.ones(n_edges, dtype=torch.bool, device=edge_index.device)
        mask[drop_id] = False
        return edge_index[:, mask], edge_type[mask]

    def drop_ui_view(
        self,
        item_attn_mean: torch.Tensor,
        inter_indices: torch.Tensor,
        inter_values: torch.Tensor
    ):
        """Rationale-guided interaction dropout for the UI-only view: an
        interaction is kept with probability proportional to its item's mean KG
        rationale score (Gumbel-noised, softmax-normalized), so items with a
        more explanatory KG neighborhood are represented more often.
        """
        edge_prob = item_attn_mean[inter_indices[1]]
        noise = -torch.log(-torch.log(torch.rand_like(edge_prob)))
        edge_prob = F.softmax(edge_prob + noise, dim=0)

        n_keep = int(self.keep_rate * inter_values.shape[0])
        sampled = torch.multinomial(edge_prob, n_keep, replacement=False)
        return inter_indices[:, sampled], inter_values[sampled] / self.keep_rate

    @staticmethod
    def _cos_sim(z1, z2):
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        return (z1 * z2).sum(dim=1)

    def forward(self, view_ui: torch.Tensor, view_kg: torch.Tensor) -> torch.Tensor:
        h_ui, h_kg = self.proj_ui(view_ui), self.proj_kg(view_kg)

        f = lambda x: torch.exp(x / self.tau)
        pos = f(self._cos_sim(h_ui, h_kg))

        perm = torch.randperm(h_ui.shape[0], device=h_ui.device)
        neg = f(self._cos_sim(h_ui, h_kg[perm])) + f(self._cos_sim(h_kg, h_ui[perm]))

        return -torch.log(pos / (2 * pos + neg)).mean()


@model_registry.register()
class KGRec(KnowledgeAwareRecommender, GraphBasedRecommender):
    """
    Knowledge Graph Self-Supervised Rationalization for Recommendation

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/3580305.3599400>`_

    Besides the main attention-weighted KG-propagation recommendation task, `KGRec`
    trains two rationale-driven self-supervised auxiliary tasks: masked-edge
    modeling (a random plus attention-selected subset of KG edges is hidden from
    the encoder, then reconstructed from its output embeddings) and a contrastive
    task between an item's KG-only and interaction-only representations, both
    augmented by dropping edges the model's own attention scores mark as
    least/most explanatory.

    Args:
        learning_rate: Learning rate
        epochs: Number of epochs
        factors: Embedding size
        batch_size: Batch size
        decay: L2 regularization weight for the user/entity embeddings
        n_layers: Number of context hops (graph convolution layers)
        node_dropout: Whether to apply node dropout
        node_dropout_rate: Node dropout ratio
        mess_dropout: Whether to apply message dropout
        mess_dropout_rate: Message dropout ratio
        mae_coef: Weight of the masked-edge-modeling loss
        mae_msize: Number of (attention-selected) KG edges masked per step
        cl_coef: Weight of the contrastive loss
        cl_tau: Contrastive loss temperature
        cl_drop_ratio: Fraction of edges dropped when building each contrastive view

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        KGRec:
          meta:
            save_recs: True
          learning_rate: 0.0001
          epochs: 50
          batch_size: 1024
          factors: 64
          n_layers: 3
          decay: 0.00001
          node_dropout: True
          node_dropout_rate: 0.5
          mess_dropout: True
          mess_dropout_rate: 0.1
          mae_coef: 0.1
          mae_msize: 256
          cl_coef: 0.01
          cl_tau: 1.0
          cl_drop_ratio: 0.5
    """

    loaders = ["KGTriplesLoader"]

    # Model hyperparameters
    factors: int = 64
    n_layers: int = 3
    learning_rate: float = 1e-4
    decay: float = 1e-5
    node_dropout: bool = True
    node_dropout_rate: float = 0.5
    mess_dropout: bool = True
    mess_dropout_rate: float = 0.1
    mae_coef: float = 0.1
    mae_msize: int = 256
    cl_coef: float = 0.01
    cl_tau: float = 1.0
    cl_drop_ratio: float = 0.5

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super(KGRec, self).__init__(params, seed, interactions, *args, **kwargs)

        # Relation-id 0 is reserved here for the "interacts" relation, so the
        # attention-based graph convolution can size its per-relation weight to
        # exclude it
        self.n_relations = self.n_relations + 1

        # KG relation graph: the (head, tail) edges plus their parallel relation type,
        # de-duplicated since relation-aware propagation expects one edge per distinct
        # triple, not per occurrence in the raw data
        edge_index = torch.stack([self.kg_heads, self.kg_tails], dim=0)
        self.edge_index, self.edge_type = dedupe_edges(edge_index, self.kg_relations, device=self._device)

        # Shift relation ids by 1 to keep id 0 reserved for "interacts"
        self.edge_type = self.edge_type + 1

        # Row-normalized (D^{-1}A) user -> entity interaction matrix
        self.interact_mat = self.build_interact_mat()

        # Embeddings
        self.user_embed = nn.Embedding(self._num_users, self.factors)
        self.entity_embed = nn.Embedding(self.n_entities, self.factors)

        self.gcn = KGAttentionGraphConv(
            channel=self.factors,
            n_hops=self.n_layers,
            n_relations=self.n_relations,
            mess_dropout_rate=self.mess_dropout_rate
        )
        self.mae = MaskedEdgeModeling(mask_size=self.mae_msize)
        self.contrast = ContrastHead(self.factors, tau=self.cl_tau, drop_ratio=self.cl_drop_ratio)

        # Node dropout over the KG edges and the user-entity interaction matrix
        self.edge_dropout = RelationAwareEdgeDropout(self.node_dropout_rate)
        self.sparse_dropout = SparseDropout(self.node_dropout_rate)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Sampler configuration
        self.sampler_config = {
            "name": "PairWiseSampler"
        }

        # Init embedding weights
        self.apply(xavier_uniform_init)

        # Move to device
        self.to(self._device)

    def forward(self, edge_index=None, edge_type=None, interact_mat=None):
        if edge_index is None:
            edge_index, edge_type = self.edge_index, self.edge_type
        if interact_mat is None:
            interact_mat = self.interact_mat

        entity_gcn_emb, user_gcn_emb = self.gcn(
            self.user_embed.weight,
            self.entity_embed.weight,
            edge_index,
            edge_type,
            interact_mat,
            mess_dropout=self.training and self.mess_dropout
        )

        return user_gcn_emb, entity_gcn_emb

    def train_step(self, batch, *args):
        user, pos, neg = [x.to(self._device) for x in batch]

        entity_emb = self.entity_embed.weight
        n_entities = entity_emb.shape[0]

        # Relation-aware node dropout over the full KG graph
        edge_index, edge_type = self.edge_index, self.edge_type
        if self.node_dropout:
            edge_index, edge_type = self.edge_dropout(edge_index, edge_type)

        # Rationale (attention) scores over the dropped-down graph, plus each
        # entity's mean rationale score (used below to guide UI edge dropout)
        edge_attn_score = self.gcn.rationale_scores(entity_emb, edge_index, edge_type)
        mean_as_head = scatter_mean(edge_attn_score, edge_index[0], dim=0, dim_size=n_entities)
        mean_as_tail = scatter_mean(edge_attn_score, edge_index[1], dim=0, dim_size=n_entities)
        mean_as_head[mean_as_head == 0.] = 1.
        mean_as_tail[mean_as_tail == 0.] = 1.
        item_attn_mean = 0.5 * mean_as_head + 0.5 * mean_as_tail

        # Rationale-guided edges masked out of the encoder graph for the
        # masked-edge-modeling task
        enc_edge_index, enc_edge_type, masked_edge_index, masked_edge_type = self.mae.split_edges(
            edge_index, edge_type, edge_attn_score
        )

        interact_mat = (
            self.sparse_dropout(self.interact_mat)
            if self.node_dropout else self.interact_mat
        )

        # Rec task: propagate over the masked-edge-modeling encoder graph
        user_gcn_emb, entity_gcn_emb = self.forward(enc_edge_index, enc_edge_type, interact_mat)

        u_e = user_gcn_emb[user]
        pos_e = entity_gcn_emb[self.item_entity_ids[pos]]
        neg_e = entity_gcn_emb[self.item_entity_ids[neg]]

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        rec_loss = F.softplus(-(pos_scores - neg_scores)).sum()

        reg_loss = self.decay * torch.mean(torch.stack([
            0.5 * u_e.pow(2).sum(), 0.5 * pos_e.pow(2).sum(), 0.5 * neg_e.pow(2).sum()
        ]))

        # Masked-edge modeling: reconstruct the masked-out edges from the
        # encoder's contextualized entity embeddings
        mae_loss = self.mae_coef * self.mae(
            entity_gcn_emb, masked_edge_index, masked_edge_type, self.gcn.relation_emb
        )

        # Contrastive task: rationale-guided KG-only vs. interaction-only item views
        cl_kg_edge_index, cl_kg_edge_type = self.contrast.drop_kg_view(
            edge_index, edge_type, edge_attn_score
        )
        inter_indices, inter_values = self.interact_mat.indices(), self.interact_mat.values()
        cl_ui_indices, cl_ui_values = self.contrast.drop_ui_view(
            item_attn_mean, inter_indices, inter_values
        )
        cl_interact_mat = torch.sparse_coo_tensor(
            cl_ui_indices, cl_ui_values, self.interact_mat.shape, device=self._device
        ).coalesce()

        item_view_kg = self.gcn.forward_kg(
            entity_emb, cl_kg_edge_index, cl_kg_edge_type, mess_dropout=False
        )
        item_view_ui = self.gcn.forward_ui(
            self.user_embed.weight, entity_emb, cl_interact_mat, mess_dropout=False
        )

        cl_loss = self.cl_coef * self.contrast(
            item_view_ui[self.item_entity_ids], item_view_kg[self.item_entity_ids]
        )

        return rec_loss + reg_loss + mae_loss + cl_loss

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
