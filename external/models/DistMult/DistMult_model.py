"""
Module description:

"""

__version__ = '0.1'
__author__ = 'Vito Walter Anelli, Claudio Pomo, Daniele Malitesta'
__email__ = 'vitowalter.anelli@poliba.it, claudio.pomo@poliba.it, daniele.malitesta@poliba.it'

import os

import numpy as np
import tensorflow as tf
from tensorflow import keras
import typing as t

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class DistMultModel(keras.Model):

    def __init__(self,
                 side,
                 learning_rate,
                 factors,
                 F2,
                 N3,
                 corruption,
                 input_type,
                 blackbox_lambda,
                 mask,
                 random_seed=42,
                 name="NNBPRMF",
                 **kwargs):
        super().__init__(name=name, **kwargs)
        tf.random.set_seed(random_seed)

        self.side = side
        self.learning_rate = learning_rate
        self.factors = factors
        self.F2 = F2
        self.N3 = N3
        self.corruption = corruption
        self.input_type = input_type
        self.blackbox_lambda = blackbox_lambda
        self.mask = mask
        init_size = 1e-3

        self.initializer = tf.initializers.GlorotUniform()

        self.entity_embeddings = keras.layers.Embedding(input_dim=self.side.nb_entities, output_dim=self.factors,
                                                        embeddings_initializer=self.initializer,
                                                        # embeddings_regularizer=keras.regularizers.l2(self.l_w),
                                                        trainable=True, dtype=tf.float32)
        self.predicate_embeddings = keras.layers.Embedding(input_dim=self.side.nb_predicates, output_dim=self.factors,
                                                           embeddings_initializer=self.initializer,
                                                           # embeddings_regularizer=keras.regularizers.l2(self.l_w),
                                                           trainable=True, dtype=tf.float32)

        self.entity_embeddings(0)
        self.predicate_embeddings(0)

        # TODO: Scale operation multiplying the embeddings with init_size

        self.optimizer = tf.optimizers.Adam(self.learning_rate)
        if self.blackbox_lambda is None:
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()
        else:
            #TODO: NegativeMRR(lambda=blackbox_lambda)
            self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy()

        #TODO: masks

    @tf.function
    def score(self,
              rel: tf.Tensor,
              arg1: tf.Tensor,
              arg2: tf.Tensor,
              *args, **kwargs) -> tf.Tensor:
        # [B]
        #TODO: check the axis and the dimensions
        res = tf.reduce_sum(rel * arg1 * arg2, 1)
        return res

    @tf.function
    def call(self,
             rel: t.Optional[tf.Tensor] = None,
             arg1: t.Optional[tf.Tensor] = None,
             arg2: t.Optional[tf.Tensor] = None,
             entity_embeddings: t.Optional[tf.Tensor] = None,
             predicate_embeddings:  t.Optional[tf.Tensor] = None,
             *args, **kwargs) -> tf.Tensor:
        # [N, E]
        ent_emb = self.entity_embeddings if self.entity_embeddings is not None else entity_embeddings
        pred_emb = self.predicate_embeddings if self.predicate_embeddings is not None else predicate_embeddings

        assert ((1 if rel is None else 0) + (1 if arg1 is None else 0) + (1 if arg2 is None else 0)) == 1

        # [B] Tensor
        scores = None

        # [B, N] = [B, E] @ [E, N]
        if rel is None:
            scores = (arg1 * arg2) @ tf.transpose(pred_emb.weights[0])
        elif arg1 is None:
            scores = (rel * arg2) @ tf.transpose(ent_emb.weights[0])
        elif arg2 is None:
            scores = (rel * arg1) @ tf.transpose(ent_emb.weights[0])

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

            if 's' in self.corruption:
                po_scores = self.call(xp_batch_emb, None, xo_batch_emb)
                # if self.mask is True:
                #     po_scores = po_scores + mask_po[xi_batch, :]

                loss += self.loss_function(xs_batch, po_scores)

            if 'o' in self.corruption:
                sp_scores = self.call(xp_batch_emb, xs_batch_emb, None)
                # if self.mask is True:
                #     sp_scores = sp_scores + mask_sp[xi_batch, :]

                loss += self.loss_function(xo_batch, sp_scores)

            if 'p' in self.corruption:
                so_scores = self.call(None, xs_batch_emb, xo_batch_emb)
                # if elf.mask is True:
                #     so_scores = so_scores + mask_so[xi_batch, :]

                loss += self.loss_function(xp_batch, so_scores)

            factors = [e for e in [xp_batch_emb, xs_batch_emb, xo_batch_emb]]

            # TODO: F2 and N3 regularization
            # if self.F2 is not None:
            #     loss += self.F2 * F2_reg(factors)
            # 
            # if self.N3 is not None:
            #     loss += self.N3 * N3_reg(factors)

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
