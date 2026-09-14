import torch
from torch import nn
from torch.nn import functional as F

from elliot.dataset import Interactions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import KnowledgeAwareRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.utils.registry import model_registry


@model_registry.register()
class CKE(KnowledgeAwareRecommender):
    """
    Collaborative Knowledge Base Embedding for Recommender Systems

    For further details, please refer to the `paper <https://dl.acm.org/doi/10.1145/2939672.2939673>`_

    Note:
        Only the KG's structural knowledge (entities/relations) is modeled here;
        the textual/visual knowledge components described in the paper are not.

    An item's scoring representation is its collaborative-filtering embedding plus
    its KG entity embedding; the KG side is additionally trained with a TransR-style
    (relation-specific projection) translational loss over `(head, relation, tail)`
    triples, ranking an observed tail above a corrupted one.

    Args:
        learning_rate: Learning rate
        epochs: Number of epochs
        factors: Embedding size for users/items/entities
        kg_factors: Embedding size of the KG's TransR relation space
        batch_size: Batch size
        reg_weights: (rec, kg) L2 regularization weights - respectively for the
            user/item/entity embeddings, and for the KG (TransR) embeddings

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        CKE:
          meta:
            save_recs: True
          learning_rate: 0.001
          epochs: 50
          batch_size: 1024
          factors: 64
          kg_factors: 64
          reg_weights: (0.01, 0.01)
    """

    loaders = ["KGTriplesLoader"]

    # Model hyperparameters
    factors: int = 64
    kg_factors: int = 64
    learning_rate: float = 1e-3
    lambda_UI: float = 2.5e-3
    lambda_vr: float = 1e-3
    lambda_M: float = 0.01

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        *args,
        **kwargs
    ):
        super(CKE, self).__init__(params, seed, interactions, *args, **kwargs)

        # Embeddings
        self.user_embedding = nn.Embedding(self._num_users, self.factors)
        self.item_embedding = nn.Embedding(self._num_items, self.factors)
        self.entity_embedding = nn.Embedding(self.n_entities, self.factors)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_factors)
        # Per-relation TransR projection matrix, from entity space to relation space
        self.trans_w = nn.Embedding(self.n_relations, self.factors * self.kg_factors)

        # Loss and optimizer
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        # Interaction sampler: one (user, pos, neg) triple per event
        self.sampler_config = {
            "name": "PairWiseSampler"
        }
        # Side-information sampler: one (head, relation, pos_tail, neg_tail) KG
        # quadruple per event, sampled entirely independently of the interactions
        # above and zipped into the same training batch
        self.side_info_sampler_config = {
            "name": "KGTriplesSampler",
            "kg_heads": self.kg_heads.tolist(),
            "kg_relations": self.kg_relations.tolist(),
            "kg_tails": self.kg_tails.tolist(),
            "n_entities": self.n_entities
        }

        # Init embedding weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def _project_to_relation_space(
        self,
        entity_ids: torch.Tensor,
        relation_ids: torch.Tensor
    ) -> torch.Tensor:
        """Project entity embeddings into each triple's own relation-specific
        (TransR) embedding space.

        Args:
            entity_ids (torch.Tensor): The `(batch,)` entity ids to project.
            relation_ids (torch.Tensor): The parallel `(batch,)` relation ids whose
                projection matrix to use.

        Returns:
            torch.Tensor: The `(batch, kg_factors)` projected embeddings.
        """
        e = self.entity_embedding(entity_ids).unsqueeze(1)
        trans_w = self.trans_w(relation_ids).view(-1, self.factors, self.kg_factors)
        return torch.bmm(e, trans_w).squeeze(1)

    def forward(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight + self.entity_embedding(self.item_entity_ids)
        return user_embeddings, item_embeddings

    def train_step(self, batch, *args):
        user, pos, neg, h, r, pos_t, neg_t = [x.to(self._device) for x in batch]

        u_e = self.user_embedding(user)
        # Keep the CF "offset" (eta_j) and the KG "structural" (v_j) parts separate
        pos_offset_e = self.item_embedding(pos)
        neg_offset_e = self.item_embedding(neg)
        pos_struct_e = self.entity_embedding(self.item_entity_ids[pos])
        neg_struct_e = self.entity_embedding(self.item_entity_ids[neg])

        pos_e = pos_offset_e + pos_struct_e
        neg_e = neg_offset_e + neg_struct_e

        pos_scores = (u_e * pos_e).sum(dim=1)
        neg_scores = (u_e * neg_e).sum(dim=1)
        rec_loss = F.softplus(-(pos_scores - neg_scores)).sum()

        # TransR projections
        h_e = self._project_to_relation_space(h, r)
        pos_t_e = self._project_to_relation_space(pos_t, r)
        neg_t_e = self._project_to_relation_space(neg_t, r)
        r_e = self.relation_embedding(r)

        # TransR: rank an observed tail (h + r ~= pos_t) above a corrupted one
        pos_tail_score = (h_e + r_e - pos_t_e).pow(2).sum(dim=1)
        neg_tail_score = (h_e + r_e - neg_t_e).pow(2).sum(dim=1)
        kg_loss = F.softplus(-(neg_tail_score - pos_tail_score)).sum()

        # --- Regularization: separate Gaussian priors ---

        # CF side (paper's lambda_U, lambda_I): user vectors + item *offset* vectors only.
        rec_reg = self.lambda_UI * 0.5 * (
            u_e.pow(2).sum() + pos_offset_e.pow(2).sum() + neg_offset_e.pow(2).sum()
        )

        # KG side (paper's lambda_v, lambda_r, lambda_M): raw entity embeddings
        # for every entity touched this step (both the explicit KG triple
        # and the items' own structural embeddings), raw relation embeddings,
        # and the per-relation projection matrices
        h_ent_e = self.entity_embedding(h)
        pos_t_ent_e = self.entity_embedding(pos_t)
        neg_t_ent_e = self.entity_embedding(neg_t)
        trans_w_r = self.trans_w(r)

        kg_reg = self.lambda_vr * 0.5 * (
            h_ent_e.pow(2).sum() + pos_t_ent_e.pow(2).sum() + neg_t_ent_e.pow(2).sum()
            + pos_struct_e.pow(2).sum() + neg_struct_e.pow(2).sum()
            + r_e.pow(2).sum()
        ) + self.lambda_M * 0.5 * trans_w_r.pow(2).sum()

        return rec_loss + kg_loss + rec_reg + kg_reg

    def predict(self, user_indices, item_indices=None, **kwargs):
        all_user_embeddings, all_item_embeddings = self.forward()
        user_embeddings = all_user_embeddings[user_indices]

        if item_indices is None:
            item_embeddings = all_item_embeddings
            einsum_string = "be,ie->bi"
        else:
            item_embeddings = all_item_embeddings[item_indices.clamp(min=0)]
            einsum_string = "be,bse->bs"

        return torch.einsum(einsum_string, user_embeddings, item_embeddings)
