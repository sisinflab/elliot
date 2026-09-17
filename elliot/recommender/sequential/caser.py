import torch
import torch.nn.functional as F
from torch import nn

from elliot.dataset import Interactions, Sessions
from elliot.namespace import RecommenderConfig
from elliot.recommender.base_recommender import SequentialRecommender
from elliot.recommender.init import xavier_normal_init
from elliot.recommender.losses import BPRLoss, EmbLoss
from elliot.utils.registry import model_registry


@model_registry.register()
class Caser(SequentialRecommender):
    """
    Caser: Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding

    For further details, please refer to the `paper <https://doi.org/10.1145/3159652.3159656>`_

    Args:
        embedding_size: Dimension of the item and user embeddings
        n_h: Number of horizontal convolutional filters
        n_v: Number of vertical convolutional filters
        dropout_prob: Dropout applied before the fully-connected layers
        reg_weight: L2 regularization weight over the embeddings used in a batch
        weight_decay: Weight decay passed to the Adam optimizer
        neg_samples: Number of negative items sampled per positive (0 disables
            negative sampling and switches to a full-catalog cross-entropy loss)
        max_seq_len: Maximum number of items kept from a user's history
        target_len: Number of consecutive future items predicted from a single context window

    To include the recommendation model, add it to the config file adopting the following pattern:

    .. code:: yaml

      models:
        Caser:
          meta:
            save_recs: True
          learning_rate: 0.001
          epochs: 50
          batch_size: 256
          embedding_size: 64
          n_h: 8
          n_v: 4
          dropout_prob: 0.5
          neg_samples: 1
          max_seq_len: 10
          target_len: 1

    Note:
        Unlike SASRec/GRU4Rec/BERT4Rec, Caser learns an explicit per-user
        embedding, so it can only be personalized correctly when its `predict`
        is given the item sequence's real owning user index. This holds for
        FLAT evaluation; under SESSION_ONLY evaluation `collector.py` resolves
        the eval row back to its owning user for exactly this reason.
    """

    # Model hyperparameters
    embedding_size: int = 64
    n_h: int = 8
    n_v: int = 4
    dropout_prob: float = 0.5
    reg_weight: float = 0.0
    weight_decay: float = 0.0
    learning_rate: float = 0.001
    neg_samples: int = 1
    max_seq_len: int = 10
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
        self.user_embedding = nn.Embedding(self._num_users, self.embedding_size)
        # Item ids [0, n_items) are real items; n_items is the padding token
        self.item_embedding = nn.Embedding(
            self._num_items + 1, self.embedding_size, padding_idx=self._num_items
        )

        # Convolutional layers
        # Vertical filters: span the whole sequence, one embedding dim at a
        # time, to capture (weighted) union-level patterns across items
        self.conv_v = nn.Conv2d(
            in_channels=1, out_channels=self.n_v, kernel_size=(self.max_seq_len, 1)
        )
        # Horizontal filters: one Conv2d per window width i in [1, max_seq_len],
        # sliding over the full embedding dim, to capture local sequential patterns
        self.conv_h = nn.ModuleList([
            nn.Conv2d(in_channels=1, out_channels=self.n_h, kernel_size=(i, self.embedding_size))
            for i in range(1, self.max_seq_len + 1)
        ])

        # Fully-connected layers
        self.fc1_dim_v = self.n_v * self.embedding_size
        self.fc1_dim_h = self.n_h * self.max_seq_len
        self.fc1 = nn.Linear(self.fc1_dim_v + self.fc1_dim_h, self.embedding_size)
        self.fc2 = nn.Linear(self.embedding_size + self.embedding_size, self.embedding_size)

        self.dropout = nn.Dropout(self.dropout_prob)
        self.ac_conv = nn.ReLU()
        self.ac_fc = nn.ReLU()

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
            "name": "UserSequentialSampler",
            "max_seq_len": self.max_seq_len,
            "neg_samples": self.neg_samples,
            "target_len": self.target_len,
        }

        # Init weights
        self.apply(xavier_normal_init)

        # Move to device
        self.to(self._device)

    def forward(self, user, item_seq):
        """Encode a batch of (user, item sequence) pairs into their next-item
        representation.

        Args:
            user (Tensor): Owning (private) user index per sequence, shape `(batch,)`.
            item_seq (Tensor): Padded item sequences, shape `(batch, max_seq_len)`.

        Returns:
            Tensor: The sequence output embedding, shape `(batch, embedding_size)`.
        """
        item_seq_emb = self.item_embedding(item_seq).unsqueeze(1)
        user_emb = self.user_embedding(user)

        out_v = self.conv_v(item_seq_emb).view(-1, self.fc1_dim_v)

        out_hs = []
        for conv in self.conv_h:
            conv_out = self.ac_conv(conv(item_seq_emb).squeeze(3))
            out_hs.append(F.max_pool1d(conv_out, conv_out.size(2)).squeeze(2))
        out_h = torch.cat(out_hs, 1)

        conv_out = self.dropout(torch.cat([out_v, out_h], 1))
        z = self.ac_fc(self.fc1(conv_out))
        seq_output = self.ac_fc(self.fc2(torch.cat([z, user_emb], 1)))
        return seq_output

    def train_step(self, batch, *args):
        batch = [x.to(self._device) for x in batch]
        if self.neg_samples > 0:
            user, item_seq, _, pos_item, neg_item = batch
        else:
            user, item_seq, _, pos_item = batch
            neg_item = None

        # Encode the (user, sequence) pair to get its next-item representation
        seq_output = self.forward(user, item_seq)

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
                self.item_embedding(item_seq), self.user_embedding(user), pos_items_emb, neg_items_emb
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
                self.item_embedding(item_seq), self.user_embedding(user), pos_items_emb
            )

        return main_loss + reg_loss

    def predict(self, user_seq, item_indices=None, user_indices=None, **kwargs):
        # `user_seq` is only padded up to the longest sequence in this batch,
        # which can be narrower than `max_seq_len`; the convolutions need the
        # fixed width they were built (and trained) for
        user_seq = self._pad_to_length(user_seq, self.max_seq_len, self._num_items).to(self._device)
        user_indices = user_indices.to(self._device)

        # Encode the (user, sequence) pair to get its next-item representation
        seq_output = self.forward(user_indices, user_seq)

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
