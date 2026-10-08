import torch
from torch import nn

from elliot.dataset import Interactions, Sessions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import SequentialRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.recommender.losses import BPRLoss, EmbLoss
from elliot.utils.registry import model_registry


@model_registry.register()
class SASRec(SequentialRecommender):
    """
    SASRec: Self-Attentive Sequential Recommendation

    For further details, please refer to the `paper <https://doi.org/10.1109/ICDM.2018.00035>`_

    Args:
        embedding_size: Dimension of the item/position embeddings
        n_layers: Number of Transformer encoder layers
        n_heads: Number of self-attention heads
        inner_size: Dimension of the feed-forward layer inside each Transformer layer
        dropout_prob: Dropout applied to the input embeddings
        attn_dropout_prob: Dropout applied inside the Transformer encoder layers
        reg_weight: L2 regularization weight over the embeddings used in a batch
        weight_decay: Weight decay passed to the Adam optimizer
        neg_samples: Number of negative items sampled per positive (0 disables
            negative sampling and switches to a full-catalog cross-entropy loss)
        max_seq_len: Maximum number of items kept from a user's history
        target_len: Number of consecutive future items predicted from a single context window

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        SASRec:
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
          neg_samples: 1
          max_seq_len: 10
          target_len: 1
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
    neg_samples: int = 1
    max_seq_len: int = 20
    target_len: int = 1

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
        self.ce_loss = nn.CrossEntropyLoss()
        self.bpr_loss = BPRLoss()
        self.reg_loss = EmbLoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        # Sampler configuration
        self.sampler_config = {
            "name": "SequentialSampler",
            "max_seq_len": self.max_seq_len,
            "neg_samples": self.neg_samples,
            "target_len": self.target_len,
        }

        # Init weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def forward(self, item_seq, item_seq_len):
        """Encode a batch of item sequences into their next-item representation.

        Args:
            item_seq (Tensor): Padded item sequences, shape `(batch, max_seq_len)`.
            item_seq_len (Tensor): True (pre-padding) sequence lengths, shape `(batch,)`.

        Returns:
            Tensor: The representation of the position right after the last real
                item of each sequence, shape `(batch, embedding_size)`.
        """
        seq_len = item_seq.size(1)
        padding_mask = item_seq == self._num_items

        position_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand_as(item_seq)
        seq_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        seq_emb = self.emb_dropout(self.layernorm(seq_emb))

        transformer_output = self.transformer_encoder(
            src=seq_emb, mask=self.causal_mask[:seq_len, :seq_len], src_key_padding_mask=padding_mask
        )
        return self._gather_indexes(transformer_output, item_seq_len - 1)

    def train_step(self, batch, *args):
        batch = [x.to(self._device) for x in batch]
        if self.neg_samples > 0:
            item_seq, item_seq_len, pos_item, neg_item = batch
        else:
            item_seq, item_seq_len, pos_item = batch
            neg_item = None

        # Encode the sequence to get its next-item representation
        seq_output = self.forward(item_seq, item_seq_len)

        # The sampler keeps `pos_item`/`neg_item` at their original scalar/
        # `(batch, neg_samples)` shape when target_len == 1
        if self.target_len == 1:
            pos_item = pos_item.unsqueeze(1)
            if neg_item is not None:
                neg_item = neg_item.unsqueeze(1)

        pos_items_emb = self.item_embedding(pos_item)

        # Calculate BPR (sampled) or cross-entropy (full-catalog) loss
        if self.neg_samples > 0:
            neg_items_emb = self.item_embedding(neg_item)
            pos_score = torch.sum(seq_output.unsqueeze(1) * pos_items_emb, dim=-1)
            neg_score = torch.sum(seq_output.unsqueeze(1).unsqueeze(1) * neg_items_emb, dim=-1)
            main_loss = self.bpr_loss(pos_score, neg_score)
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq), pos_items_emb, neg_items_emb
            )
        else:
            logits = torch.matmul(
                seq_output, self.item_embedding.weight.transpose(0, 1)
            )
            # Every one of the target_len positives is predicted from the same
            # context embedding, so it's the same logits scored against each
            logits = logits.unsqueeze(1).expand(-1, self.target_len, -1).reshape(-1, logits.size(-1))
            main_loss = self.ce_loss(logits, pos_item.reshape(-1))
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(item_seq), pos_items_emb
            )

        return main_loss + reg_loss

    def predict(self, user_seq, seq_len, item_indices=None, user_indices=None, **kwargs):
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)

        # Encode the sequence to get its next-item representation
        seq_output = self.forward(user_seq, seq_len)

        # Compute predictions
        if item_indices is None:
            item_embeddings = self.item_embedding.weight[:-1, :]
            einsum_string = "be,ie->bi"
        else:
            item_indices = item_indices.to(self._device)
            item_embeddings = self.item_embedding(item_indices.clamp(min=0))
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(einsum_string, seq_output, item_embeddings)
        return predictions
