"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Pasquale Minervini'
__email__ = 'p.minervini@ucl.ac.uk'

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import typing as t

from .regularizers import F2, N3

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class ComplExModel(keras.Model):
    def __init__(self,
                 side,
                 learning_rate,
                 factors,
                 F2_weight,
                 N3_weight,
                 input_type,
                 random_seed=42,
                 name="NNBPRMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.side = side
        self.learning_rate = learning_rate
        self.factors = factors
        self.F2_weight = F2_weight
        self.N3_weight = N3_weight
        self.input_type = input_type

        init_size = 1e-3

        self.initializer = tf.initializers.GlorotUniform()

        self.entity_embeddings = keras.layers.Embedding(input_dim=self.side.nb_entities, output_dim=self.factors,
                                                        embeddings_initializer=self.initializer,
                                                        trainable=True, dtype=tf.float32)
        self.predicate_embeddings = keras.layers.Embedding(input_dim=self.side.nb_predicates, output_dim=self.factors,
                                                           embeddings_initializer=self.initializer,
                                                           trainable=True, dtype=tf.float32)

        # XXX: Why these?
        self.entity_embeddings(0)
        self.predicate_embeddings(0)

        # TODO: Scale operation multiplying the embeddings with init_size
        self.optimizer = tf.optimizers.Adagrad(self.learning_rate)

        # Is this the correct loss function to use? Why "sparse"? Check with Walter.
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        self.F2_reg = F2()
        self.N3_reg = N3()


    @tf.function
    def score(self,
              rel: tf.Tensor,
              arg1: tf.Tensor,
              arg2: tf.Tensor,
              *args, **kwargs) -> tf.Tensor:
        # [B]
        batch_size = rel.shape[0]
        embedding_size = rel.shape[1]

        assert rel.shape[0] == arg1.shape[0] == arg2.shape[0] == batch_size
        assert rel.shape[1] == arg1.shape[1] == arg2.shape[1] == embedding_size

        rank = embedding_size // 2

        # [B, E]
        rel_real, rel_img = rel[:, :rank], rel[:, rank:]
        arg1_real, arg1_img = arg1[:, :rank], arg1[:, rank:]
        arg2_real, arg2_img = arg2[:, :rank], arg2[:, rank:]

        # [B] Tensor
        res = tf.reduce_sum(rel_real * arg1_real * arg2_real +
                            rel_real * arg1_img * arg2_img +
                            rel_img * arg1_real * arg2_img -
                            rel_img * arg1_img * arg2_real, 1)

        return res

    @tf.function
    def call(self,
             rel: t.Optional[tf.Tensor] = None,
             arg1: t.Optional[tf.Tensor] = None,
             arg2: t.Optional[tf.Tensor] = None,
             entity_embeddings: t.Optional[tf.Tensor] = None,
             *args, **kwargs) -> tf.Tensor:
        # [N, E]
        ent_emb = self.entity_embeddings if self.entity_embeddings is not None else entity_embeddings

        assert ((1 if arg1 is None else 0) + (1 if arg2 is None else 0)) == 1

        # [B] Tensor
        scores = None

        batch_size = rel.shape[0]
        embedding_size = rel.shape[1]

        rank = embedding_size // 2

        ent_real, ent_img = ent_emb.embeddings[:, :rank], ent_emb.embeddings[:, rank:]

        # [B, N] = [B, E] @ [E, N]
        if arg1 is None:
            rel_real, rel_img = rel[:, :rank], rel[:, rank:]
            arg2_real, arg2_img = arg2[:, :rank], arg2[:, rank:]

            score1_po = (rel_real * arg2_real + rel_img * arg2_img) @ tf.transpose(ent_real)
            score2_po = (rel_real * arg2_img - rel_img * arg2_real) @ tf.transpose(ent_img)

            scores = score1_po + score2_po

        elif arg2 is None:
            rel_real, rel_img = rel[:, :rank], rel[:, rank:]
            arg1_real, arg1_img = arg1[:, :rank], arg1[:, rank:]

            score1_sp = (rel_real * arg1_real - rel_img * arg1_img) @ tf.transpose(ent_real)
            score2_sp = (rel_real * arg1_img + rel_img * arg1_real) @ tf.transpose(ent_img)

            scores = score1_sp + score2_sp

        assert scores is not None

        return scores

    @tf.function
    def train_step(self, batch):
        with tf.GradientTape() as tape:
            xp_batch, xs_batch, xo_batch, xi_batch = batch

            xp_batch_emb = self.predicate_embeddings(xp_batch)
            xs_batch_emb = self.entity_embeddings(xs_batch)
            xo_batch_emb = self.entity_embeddings(xo_batch)

            loss = 0.0

            po_scores = self.call(xp_batch_emb, None, xo_batch_emb)
            loss += self.loss_function(xs_batch, po_scores)

            sp_scores = self.call(xp_batch_emb, xs_batch_emb, None)
            loss += self.loss_function(xo_batch, sp_scores)

            factors = [e for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

            # TODO: F2 and N3 regularization
            if self.F2_weight is not None:
                loss += self.F2_weight * self.F2_reg(factors)

            if self.N3_weight is not None:
                loss += self.N3_weight * self.N3_reg(factors)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss

    @tf.function
    def predict(self, xp_batch=None, xs_batch=None, xo_batch=None, training=False, **kwargs):
        xp_batch_emb = self.predicate_embeddings(xp_batch) if xp_batch is not None else None
        xs_batch_emb = self.entity_embeddings(xs_batch) if xs_batch is not None else None
        xo_batch_emb = self.entity_embeddings(xo_batch) if xo_batch is not None else None
        scores = self.call(xp_batch_emb, xs_batch_emb, xo_batch_emb)
        return scores

    @tf.function
    def get_top_k(self, preds, train_mask, k=100):
        return tf.nn.top_k(tf.where(train_mask, preds, -np.inf), k=k, sorted=True)

    def get_config(self):
        raise NotImplementedError
