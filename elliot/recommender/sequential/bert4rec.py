import torch
from torch import nn

from elliot.dataset import Interactions, Sessions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import SequentialRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.recommender.losses import BPRLoss, EmbLoss
from elliot.utils.registry import model_registry


@model_registry.register()
class BERT4Rec(SequentialRecommender):
    """
    BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer

    For further details, please refer to the `paper <https://doi.org/10.1145/3357384.3357895>`_

    This model uses a bidirectional Transformer trained on a masked item
    prediction (Cloze) task. A `[MASK]` token is appended to the sequence at
    prediction time to represent the next, yet-unseen item.

    Args:
        embedding_size: Dimension of the item/position embeddings
        n_layers: Number of Transformer encoder layers
        n_heads: Number of self-attention heads
        inner_size: Dimension of the feed-forward layer inside each Transformer layer
        dropout_prob: Dropout applied to the input embeddings
        attn_dropout_prob: Dropout applied inside the Transformer encoder layers
        mask_prob: Fraction of a training window's items replaced by `[MASK]`
        reg_weight: L2 regularization weight over the embeddings used in a batch
        weight_decay: Weight decay passed to the Adam optimizer
        neg_samples: Number of negative items sampled per masked position (0 disables
            negative sampling and switches to a full-catalog cross-entropy loss)
        max_seq_len: Maximum number of items kept from a user's history

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        BERT4Rec:
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
          mask_prob: 0.2
          neg_samples: 1
          max_seq_len: 50
    """

    # Model hyperparameters
    embedding_size: int = 64
    n_layers: int = 2
    n_heads: int = 2
    inner_size: int = 256
    dropout_prob: float = 0.2
    attn_dropout_prob: float = 0.2
    mask_prob: float = 0.2
    reg_weight: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 0.001
    neg_samples: int = 0
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

        # Item ids [0, n_items) are real items; n_items is the padding token
        # (shared with every other sequential sampler); n_items + 1 is [MASK]
        self.padding_token_id = self._num_items
        self.mask_token_id = self._num_items + 1

        # Embeddings
        self.item_embedding = nn.Embedding(
            self._num_items + 2, self.embedding_size, padding_idx=self.padding_token_id
        )
        # Training windows only ever span positions [0, max_seq_len),
        # so at prediction time the context is cut to max_seq_len - 1 items
        # to keep the appended [MASK] on a trained position
        self.position_embedding = nn.Embedding(self.max_seq_len, self.embedding_size)
        self.layernorm = nn.LayerNorm(self.embedding_size, eps=1e-8)
        self.dropout = nn.Dropout(self.dropout_prob)

        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embedding_size,
            nhead=self.n_heads,
            dim_feedforward=self.inner_size,
            dropout=self.attn_dropout_prob,
            activation="gelu",
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=self.n_layers,
            enable_nested_tensor=False
        )

        # Per-item output bias, as in the original implementation
        self.out_bias = nn.Parameter(torch.zeros(self._num_items + 1))

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
            "name": "ClozeSampler",
            "max_seq_len": self.max_seq_len,
            "neg_samples": self.neg_samples,
            "mask_prob": self.mask_prob,
            "mask_token_id": self.mask_token_id,
        }

        # Init weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def forward(self, item_seq):
        """Encode a (possibly masked) batch of item sequences with the
        bidirectional Transformer encoder.

        Args:
            item_seq (Tensor): Item sequences, shape `(batch, seq_len)`, with
                some positions possibly replaced by `self.mask_token_id`.

        Returns:
            Tensor: Per-position hidden states, shape `(batch, seq_len, embedding_size)`.
        """
        seq_len = item_seq.size(1)
        padding_mask = item_seq == self.padding_token_id

        position_ids = torch.arange(seq_len, device=item_seq.device).unsqueeze(0).expand_as(item_seq)

        input_emb = self.item_embedding(item_seq) + self.position_embedding(position_ids)
        input_emb = self.dropout(self.layernorm(input_emb))

        # Bidirectional: no causal mask
        return self.transformer_encoder(src=input_emb, mask=None, src_key_padding_mask=padding_mask)

    def train_step(self, batch, *args):
        masked_seq, pos_items, neg_items, masked_indices = [x.to(self._device) for x in batch]

        # Encode the masked sequence and gather the masked positions' output
        transformer_output = self.forward(masked_seq)
        seq_output = self._gather_multi_indexes(transformer_output, masked_indices)

        # `pos_items`/`neg_items`/`masked_indices` are fixed-size (`max_seq_len`)
        # tensors, only the first `num_to_mask` slots of a given row are real
        # masked positions; the rest are filled with `padding_token_id`
        loss_mask = pos_items != self.padding_token_id
        seq_output = seq_output[loss_mask]
        pos_items = pos_items[loss_mask]

        pos_items_emb = self.item_embedding(pos_items)

        # Calculate BPR (sampled) or cross-entropy (full-catalog) loss
        if self.neg_samples > 0:
            neg_items = neg_items[loss_mask]
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1) + self.out_bias[pos_items]
            neg_score = torch.sum(seq_output.unsqueeze(1) * neg_items_emb, dim=-1) + self.out_bias[neg_items]
            main_loss = self.bpr_loss(pos_score, neg_score)
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(masked_seq), pos_items_emb, neg_items_emb
            )
        else:
            logits = torch.matmul(
                seq_output, self.item_embedding.weight[:self._num_items].transpose(0, 1)
            ) + self.out_bias[:self._num_items]
            main_loss = self.ce_loss(logits, pos_items)
            reg_loss = self.reg_weight * self.reg_loss(
                self.item_embedding(masked_seq), pos_items_emb
            )

        return main_loss + reg_loss

    def _append_mask_token(self, user_seq, seq_len):
        """Append a `[MASK]` token right after each sequence's real context,
        to represent the (yet unseen) next item to predict.

        Contexts are first cut to their most recent `max_seq_len - 1` items,
        so that the `[MASK]` always lands on a position seen during training.

        Args:
            user_seq (Tensor): Left-aligned padded item sequences, shape
                `(batch, width)` with `width <= max_seq_len`.
            seq_len (Tensor): True (pre-padding) context lengths, shape `(batch,)`.

        Returns:
            Tuple[Tensor, Tensor]: The extended sequences, shape
                `(batch, min(width + 1, max_seq_len))`, and the position of
                each row's `[MASK]` token, shape `(batch,)`.
        """
        # Guarantee at least one column to gather from (all-empty contexts)
        user_seq = self._pad_to_length(user_seq, max(user_seq.size(1), 1), self.padding_token_id)
        batch_size, width = user_seq.shape
        out_width = min(width + 1, self.max_seq_len)
        mask_pos = seq_len.clamp(max=self.max_seq_len - 1)

        # Drop the oldest items of contexts too long to fit the [MASK] after them
        shift = (seq_len - mask_pos).unsqueeze(1)
        gather_idx = torch.arange(out_width, device=user_seq.device).unsqueeze(0) + shift
        valid = gather_idx < seq_len.unsqueeze(1)
        pred_seq = torch.where(
            valid,
            user_seq.gather(1, gather_idx.clamp(max=width - 1)),
            torch.full_like(gather_idx, self.padding_token_id),
        )

        batch_indices = torch.arange(batch_size, device=user_seq.device)
        pred_seq[batch_indices, mask_pos] = self.mask_token_id
        return pred_seq, mask_pos

    def predict(self, user_seq, seq_len, item_indices=None, user_indices=None, **kwargs):
        user_seq = user_seq.to(self._device)
        seq_len = seq_len.to(self._device)

        # Append [MASK] and encode the sequence to get its next-item representation
        pred_seq, mask_pos = self._append_mask_token(user_seq, seq_len)
        transformer_output = self.forward(pred_seq)
        seq_output = self._gather_indexes(transformer_output, mask_pos)

        # Compute predictions
        if item_indices is None:
            item_embeddings = self.item_embedding.weight[:self._num_items, :]
            bias = self.out_bias[:self._num_items]
            einsum_string = "be,ie->bi"
        else:
            item_indices = item_indices.to(self._device)
            clamped_indices = item_indices.clamp(min=0)
            item_embeddings = self.item_embedding(clamped_indices)
            bias = self.out_bias[clamped_indices]
            einsum_string = "be,bse->bs"

        predictions = torch.einsum(einsum_string, seq_output, item_embeddings) + bias
        return predictions
