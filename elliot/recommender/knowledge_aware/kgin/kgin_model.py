'''
Created on July 1, 2020
PyTorch Implementation of KGIN
@author: Tinglin Huang (tinglin.huang@zju.edu.cn)
'''
__author__ = "huangtinglin"

import random

from tensorflow import keras
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean


class Aggregator:
    """
    Relational Path-aware Convolution Network
    """
    def __init__(self, n_users, n_factors):
        super().__init__()
        self.n_users = n_users
        self.n_factors = n_factors

    @tf.function
    def call(self, entity_emb, user_emb, latent_emb,
                edge_index, edge_type, interact_mat,
                weight, disen_weight_att):

        n_entities = entity_emb.shape[0]
        channel = entity_emb.shape[1]
        n_users = self.n_users
        n_factors = self.n_factors

        """KG aggregate"""
        head, tail = edge_index
        edge_relation_emb = tf.gather(weight, edge_type - 1)  # exclude interact, remap [1, n_relations) to [0, n_relations-1)
        neigh_relation_emb = tf.gather(entity_emb, tail) * edge_relation_emb  # [-1, channel]
        entity_agg = scatter_mean(src=neigh_relation_emb, index=head, dim_size=n_entities, dim=0)

        """cul user->latent factor attention"""
        score_ = tf.matmul(user_emb, tf.transpose(latent_emb))
        score = tf.expand_dims(tf.nn.softmax(score_, axis=1), axis=-1)  # [n_users, n_factors, 1]

        """user aggregate"""
        user_agg = tf.matmul(interact_mat, entity_emb, a_is_sparse=True, b_is_sparse=True)  # [n_users, channel]
        disen_weight = tf.broadcast_to(tf.matmul(tf.nn.softmax(disen_weight_att, axis=1),
                                weight), [n_users, n_factors, channel])
        user_agg = user_agg * tf.reduce_sum((disen_weight * score), axis=1) + user_agg  # [n_users, channel]

        return entity_agg, user_agg


class GraphConv:
    """
    Graph Convolutional Network
    """
    def __init__(self, channel, n_hops, n_users,
                 n_factors, n_relations, interact_mat,
                 ind, node_dropout_rate=0.5, mess_dropout_rate=0.1):
        super(GraphConv, self).__init__()

        # self.convs = nn.ModuleList()    # TODO: Che cazz ie?
        # TODO: Controlla che la sosituzione con una lista semplice sia efficace in TF
        self.convs = []
        self.interact_mat = interact_mat
        self.n_relations = n_relations
        self.n_users = n_users
        self.n_factors = n_factors
        self.node_dropout_rate = node_dropout_rate
        self.mess_dropout_rate = mess_dropout_rate
        self.ind = ind

        self.temperature = 0.2

        self.initializer = tf.initializers.GlorotUniform()

        self.weight = tf.Variable(self.initializer(shape=[n_relations - 1, channel]))
        self.disen_weight_att = tf.Variable(self.initializer(shape=[n_factors, n_relations - 1]))

        for i in range(n_hops):
            self.convs.append(Aggregator(n_users=n_users, n_factors=n_factors))

        self.dropout = tf.keras.layers.Dropout(p=mess_dropout_rate)  # mess dropout

    def _edge_sampling(self, edge_index, edge_type, rate=0.5):
        # edge_index: [2, -1]
        # edge_type: [-1]
        n_edges = edge_index.shape[1]
        random_indices = np.random.choice(n_edges, size=int(n_edges * rate), replace=False)
        return edge_index[:, random_indices], edge_type[random_indices]

    @tf.function
    def _sparse_dropout(self, x, rate=0.5):
        noise_shape = len(x.values)

        random_tensor = rate
        random_tensor += tf.random.uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), tf.bool)
        i = x.indices
        v = x.values

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = tf.SparseTensor(i, v, x.shape)
        return out * tf.constant(1. / (1 - rate))

    @tf.function
    def _cul_cor(self):
        def cosine_similarity(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            normalized_tensor_1 = tensor_1 / tf.norm(tensor_1, axis=0, keepdims=True)
            normalized_tensor_2 = tensor_2 / tf.norm(tensor_2, axis=0, keepdims=True)
            return tf.reduce_sum(normalized_tensor_1 * normalized_tensor_2, axis=0) ** 2  # no negative

        def distance_correlation(tensor_1, tensor_2):
            # tensor_1, tensor_2: [channel]
            # ref: https://en.wikipedia.org/wiki/Distance_correlation
            channel = tensor_1.shape[0]
            zeros = tf.zeros([channel, channel])
            zero = tf.zeros(1)
            tensor_1, tensor_2 = tf.expand_dims(tensor_1, -1), tf.expand_dims(tensor_2, -1)
            """cul distance matrix"""
            a_, b_ = tensor_1 @ tf.transpose(tensor_1) * 2, \
                   tensor_2 @ tf.transpose(tensor_2) * 2  # [channel, channel]
            tensor_1_square, tensor_2_square = tensor_1 ** 2, tensor_2 ** 2
            a, b = tf.sqrt(tf.reduce_max(tensor_1_square - a_ + tf.transpose(tensor_1_square), zeros) + 1e-8), \
                   tf.sqrt(tf.reduce_max(tensor_2_square - b_ + tf.transpose(tensor_2_square), zeros) + 1e-8)  # [channel, channel]
            """cul distance correlation"""
            A = a - tf.reduce_mean(a, axis=0, keepdims=True) - tf.reduce_mean(a, axis=1, keepdims=True) + tf.reduce_mean(a)
            B = b - tf.reduce_mean(b, axis=0, keepdims=True) - tf.reduce_mean(b, axis=1, keepdims=True) + tf.reduce_mean(b)
            dcov_AB = tf.sqrt(tf.reduce_max(tf.reduce_sum(A * B) / channel ** 2, zero) + 1e-8)
            dcov_AA = tf.sqrt(tf.reduce_max(tf.reduce_sum(A * A) / channel ** 2, zero) + 1e-8)
            dcov_BB = tf.sqrt(tf.reduce_max(tf.reduce_sum(B * B) / channel ** 2, zero) + 1e-8)
            return dcov_AB / tf.sqrt(dcov_AA * dcov_BB + 1e-8)

        def mutual_information():
            # disen_T: [num_factor, dimension]
            disen_T = tf.transpose(self.disen_weight_att)

            # normalized_disen_T: [num_factor, dimension]
            normalized_disen_T = disen_T / tf.norm(disen_T, axis=1, keepdims=True)

            pos_scores = tf.reduce_sum(normalized_disen_T * normalized_disen_T, axis=1)
            ttl_scores = tf.reduce_sum(tf.matmul(disen_T, self.disen_weight_att), axis=1)

            pos_scores = tf.exp(pos_scores / self.temperature)
            ttl_scores = tf.exp(ttl_scores / self.temperature)

            mi_score = - tf.reduce_sum(tf.math.log(pos_scores / ttl_scores))
            return mi_score

        """cul similarity for each latent factor weight pairs"""
        if self.ind == 'mi':
            return mutual_information()
        else:
            cor = 0
            for i in range(self.n_factors):
                for j in range(i + 1, self.n_factors):
                    if self.ind == 'distance':
                        cor += distance_correlation(self.disen_weight_att[i], self.disen_weight_att[j])
                    else:
                        cor += cosine_similarity(self.disen_weight_att[i], self.disen_weight_att[j])
        return cor

    @tf.function
    def call(self, user_emb, entity_emb, latent_emb, edge_index, edge_type,
                interact_mat, mess_dropout=True, node_dropout=False):

        """node dropout"""
        if node_dropout:
            edge_index, edge_type = self._edge_sampling(edge_index, edge_type, self.node_dropout_rate)
            interact_mat = self._sparse_dropout(interact_mat, self.node_dropout_rate)

        entity_res_emb = entity_emb  # [n_entity, channel]
        user_res_emb = user_emb  # [n_users, channel]
        cor = self._cul_cor()
        for i in range(len(self.convs)):
            entity_emb, user_emb = self.convs[i].call(entity_emb, user_emb, latent_emb,
                                                 edge_index, edge_type, interact_mat,
                                                 self.weight, self.disen_weight_att)

            """message dropout"""
            if mess_dropout:
                entity_emb = self.dropout(entity_emb)
                user_emb = self.dropout(user_emb)
            entity_emb = tf.linalg.normalize(entity_emb)
            user_emb = tf.linalg.normalize(user_emb)

            """result emb"""
            entity_res_emb = tf.add(entity_res_emb, entity_emb)
            user_res_emb = tf.add(user_res_emb, user_emb)

        return entity_res_emb, user_res_emb, cor


class KGINModel(keras.Model):
    def __init__(self,
                 adj_mat,
                 graph,
                 item_factors,
                 learning_rate=0.001,
                 l_w=0, l_b=0,
                 random_seed=42,
                 name="KGINModel",
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.adj_mat = adj_mat
        self.graph = graph
        self.edge_index, self.edge_type = self._get_edges(graph)

        self.initializer = tf.initializers.GlorotUniform()
        self.all_embed = tf.Variable(self.initializer(shape=[self.n_nodes, self.emb_size]))
        self.latent_emb = tf.Variable(self.initializer(shape=[self.n_factors, self.emb_size]))

        coo = self.adj_mat.tocoo()
        i = np.mat([coo.row, coo.col]).T
        v = coo.data
        self.interact_mat = tf.SparseTensor(i, v, coo.shape)

        self.gcn = GraphConv(channel=self.emb_size,
                             n_hops=self.context_hops,
                             n_users=self.n_users,
                             n_relations=self.n_relations,
                             n_factors=self.n_factors,
                             interact_mat=self.interact_mat,
                             ind=self.ind,
                             node_dropout_rate=self.node_dropout_rate,
                             mess_dropout_rate=self.mess_dropout_rate)


    def _get_indices(self, X):
        coo = X.tocoo()
        return tf.transpose(tf.constant([coo.row, coo.col], dtype=tf.int32))  # [-1, 2]

    def _get_edges(self, graph):
        graph_tensor = tf.constant(list(graph.edges))  # [-1, 3]
        index = graph_tensor[:, :-1]  # [-1, 2]
        type = graph_tensor[:, -1]  # [-1, 1]
        return tf.cast(tf.transpose(index), dtype=tf.int32), tf.cast(type, dtype=tf.int32)

    @tf.function
    def call(self, batch=None):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']

        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        # entity_gcn_emb: [n_entity, channel]
        # user_gcn_emb: [n_users, channel]
        entity_gcn_emb, user_gcn_emb, cor = self.gcn.call(user_emb,
                                                          item_emb,
                                                          self.latent_emb,
                                                          self.edge_index,
                                                          self.edge_type,
                                                          self.interact_mat,
                                                          mess_dropout=self.mess_dropout,
                                                          node_dropout=self.node_dropout)
        u_e = user_gcn_emb[user]
        pos_e, neg_e = entity_gcn_emb[pos_item], entity_gcn_emb[neg_item]
        return self.create_bpr_loss(u_e, pos_e, neg_e, cor)

    @tf.function
    def train_step(self, batch):
        user, pos, neg = batch
        with tf.GradientTape() as tape:
            # Clean Inference
            xu_pos = self(inputs=(user, pos), training=True)
            xu_neg = self(inputs=(user, neg), training=True)

            difference = tf.clip_by_value(xu_pos - xu_neg, -80.0, 1e8)
            loss = tf.reduce_sum(tf.nn.softplus(-difference))
            # Regularization Component
            # reg_loss = self._l_w * tf.reduce_sum([tf.nn.l2_loss(self.H),
            #                                      tf.nn.l2_loss(self.G)]) \
            #            + self._l_b * tf.nn.l2_loss(self.B)
            #            # + self._l_b * tf.nn.l2_loss(beta_neg) / 10

            # Loss to be optimized
            # loss += reg_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return loss


    @tf.function
    def generate(self):
        user_emb = self.all_embed[:self.n_users, :]
        item_emb = self.all_embed[self.n_users:, :]
        return self.gcn.call(user_emb,
                             item_emb,
                             self.latent_emb,
                             self.edge_index,
                             self.edge_type,
                             self.interact_mat,
                             mess_dropout=False, node_dropout=False)[:-1]

    def rating(self, u_g_embeddings, i_g_embeddings):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, users, pos_items, neg_items, cor):
        batch_size = users.shape[0]
        pos_scores = torch.sum(torch.mul(users, pos_items), axis=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), axis=1)

        mf_loss = -1 * torch.mean(nn.LogSigmoid()(pos_scores - neg_scores))

        # cul regularizer
        regularizer = (torch.norm(users) ** 2
                       + torch.norm(pos_items) ** 2
                       + torch.norm(neg_items) ** 2) / 2
        emb_loss = self.decay * regularizer / batch_size
        cor_loss = self.sim_decay * cor

        return mf_loss + emb_loss + cor_loss, mf_loss, emb_loss, cor