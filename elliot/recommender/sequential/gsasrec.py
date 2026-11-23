import torch
from torch import nn

from elliot.dataset import Interactions, Sessions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import SequentialRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.recommender.losses import EmbLoss, GBCELoss
from elliot.utils.registry import model_registry


@model_registry.register()
class gSASRec(SequentialRecommender):
    """
    gSASRec: Reducing Overconfidence in Sequential Recommendation Trained with Negative Sampling

    For further details, please refer to the `paper <https://doi.org/10.1145/3604915.3608783>`_

    SASRec trained on the sequence shifting task (every position predicts its
    next item) with many uniformly sampled negatives and the gBCE loss, which
    counters the overconfidence that negative sampling induces in plain BCE.

    Args:
        embedding_size: Dimension of the item/position embeddings
        n_layers: Number of Transformer encoder layers
        n_heads: Number of self-attention heads
        inner_size: Dimension of the feed-forward layer inside each Transformer layer
        dropout_prob: Dropout applied to the input embeddings
        attn_dropout_prob: Dropout applied inside the Transformer encoder layers
        reg_weight: L2 regularization weight over the embeddings used in a batch
        weight_decay: Weight decay passed to the Adam optimizer
        neg_samples: Number of negative items sampled per positive (at least 1)
        gbce_t: gBCE calibration parameter in [0, 1]: 0 is the regular BCE loss,
            1 the fully calibrated gBCE
        reuse_item_embeddings: Whether to score items with the input item
            embeddings, instead of a separate output embedding table
        max_seq_len: Maximum number of items kept from a user's history

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        gSASRec:
          meta:
            save_recs: True
          learning_rate: 0.001
          epochs: 50
          batch_size: 256
          embedding_size: 64
          n_layers: 2
          n_heads: 2
          inner_size: 256
          dropout_prob: 0.2
          attn_dropout_prob: 0.2
          neg_samples: 256
          gbce_t: 0.75
          reuse_item_embeddings: False
          max_seq_len: 20
    """

    # Model hyperparameters
    embedding_size: int = 64
    n_layers: int = 2
    n_heads: int = 2
    inner_size: int = 256
    dropout_prob: float = 0.2
    attn_dropout_prob: float = 0.2
    reg_weight: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 0.001
    neg_samples: int = 256
    gbce_t: float = 0.75
    reuse_item_embeddings: bool = False
    max_seq_len: int = 20

    def __init__(
        self,
        params: RecommenderConfig,
        seed: int,
        interactions: Interactions,
        sessions: Sessions,
        *args,
        **kwargs
    ):
        super().__init__(params, seed, interactions, sessions, *args, **kwargs)

        # Embeddings
        # Item ids [0, n_items) are real items; n_items is the padding token
        self.item_embedding = nn.Embedding(
            self._num_items + 1, self.embedding_size, padding_idx=self._num_items
        )
        self.output_embedding = self.item_embedding if self.reuse_item_embeddings else nn.Embedding(
            self._num_items + 1, self.embedding_size, padding_idx=self._num_items
        )
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            enable_nested_tensor=False
        )

        # Additive causal mask: position i may only attend to positions <= i
        causal_mask = nn.Transformer.generate_square_subsequent_mask(self.max_seq_len).bool()
        self.register_buffer("causal_mask", causal_mask)

        # Losses
        self.gbce_loss = GBCELoss(self.neg_samples, self._num_items, self.gbce_t)
        self.reg_loss = EmbLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Sampler configuration
        # One window per user, one item longer than the model input: shifted by one,
        # it gives `max_seq_len` input positions, each with its next item as target
        self.sampler_config = {
            "name": "SlidingWindowSampler",
            "max_seq_len": self.max_seq_len + 1,
            "neg_samples": self.neg_samples,
            "stride": 1,
            "max_windows": 1,
            "exclude_seen": False,
        }

        # Init weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def forward(self, item_seq):
        """Encode a batch of item sequences with the causal Transformer encoder.

        Args:
            item_seq (Tensor): Left-aligned padded item sequences, shape `(batch, seq_len)`.

        Returns:
            Tensor: Per-position hidden states, each the representation of the
                item following that position, shape `(batch, seq_len, embedding_size)`.
        """
        seq_len = item_seq.size(1)
        padding_mask = item_seq == self._num_items

        position_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        seq_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        seq_emb = self.emb_dropout(self.layernorm(seq_emb))

        return self.transformer_encoder(
            src=seq_emb, mask=self.causal_mask[:seq_len, :seq_len], src_key_padding_mask=padding_mask
        )

    def train_step(self, batch, *args):
        pos_seq, neg_seq = [x.to(self._device) for x in batch]

        # Sequence shifting: every input position predicts the item right after it
        item_seq = pos_seq[:, :-1]
        pos_item = pos_seq[:, 1:]
        neg_item = neg_seq[:, 1:]

        # Encode the sequence to get every position's next-item representation
        seq_output = self.forward(item_seq)

        # Windows shorter than `max_seq_len + 1` are right-padded: drop their padding targets
        loss_mask = pos_item != self._num_items
        seq_output = seq_output[loss_mask]
        pos_item = pos_item[loss_mask]
        neg_item = neg_item[loss_mask]

        pos_items_emb = self.output_embedding(pos_item)
        neg_items_emb = self.output_embedding(neg_item)

        # Calculate gBCE loss
        pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)
        neg_score = torch.bmm(neg_items_emb, seq_output.unsqueeze(-1)).squeeze(-1)
        loss = self.gbce_loss(pos_score, neg_score)

        if self.reg_weight > 0:
            loss = loss + self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq), pos_items_emb, neg_items_emb
            )

        return loss

    def predict(self, user_seq, seq_len, item_indices=None, user_indices=None, **kwargs):
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)

        # Encode the sequence and take its last real position's next-item representation
        seq_output = self._gather_indexes(self.forward(user_seq), seq_len - 1)

        # Compute predictions
        if item_indices is None:
            item_embeddings = self.output_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            item_indices = item_indices.to(self._device)
            item_embeddings = self.output_embedding(item_indices.clamp(min=0))
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(einsum_string, seq_output, item_embeddings)
        return predictions
