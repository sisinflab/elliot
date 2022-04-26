from abc import ABC

from torch_geometric.nn import LGConv
from NodeNodeTextLayer import NodeNodeTextLayer
import random
import numpy as np

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageOps
import torch
import pandas as pd
import torch_geometric

from abc import ABC

import torch
from torch_geometric.nn import MessagePassing


class DGCFLayer(MessagePassing, ABC):
    def __init__(self):
        super(DGCFLayer, self).__init__(aggr='add', node_dim=-3)

    @staticmethod
    def weighted_degree(index, weights, num_nodes, dtype):
        out = torch.zeros((weights.shape[0], num_nodes,), dtype=dtype, device=weights.device)
        return out.scatter_add_(1, index.repeat(weights.shape[0], 1), weights)

    def forward(self, x, edge_index, edge_index_intents):
        normalized_edge_index_intents = torch.softmax(edge_index_intents, dim=0)
        row, col = edge_index
        deg_row = self.weighted_degree(index=row, weights=normalized_edge_index_intents, num_nodes=x.size(0),
                                       dtype=x.dtype)
        deg_col = self.weighted_degree(index=col, weights=normalized_edge_index_intents, num_nodes=x.size(0),
                                       dtype=x.dtype)
        deg = deg_row + deg_col
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[:, row] * deg_inv_sqrt[:, col]
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return torch.unsqueeze(norm.permute(1, 0), -1) * x_j


class DGCFModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 n_layers,
                 name="DGCF",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_layers = n_layers

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        dgcf_network_list = []
        for layer in range(self.n_layers):
            dgcf_network_list.append((DGCFLayer(), 'x, edge_index -> x'))

        self.dgcf_network = torch_geometric.nn.Sequential('x, edge_index', dgcf_network_list)
        self.dgcf_network.to(self.device)
        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class LightGCNModel(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 n_layers,
                 name="LightGCN",
                 **kwargs
                 ):
        super().__init__()

        # set seed
        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.weight_size_list = [self.embed_k] * (self.n_layers + 1)

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        propagation_network_list = []

        for layer in range(self.n_layers):
            propagation_network_list.append((LGConv(), 'x, edge_index -> x'))

        self.propagation_network = torch_geometric.nn.Sequential('x, edge_index', propagation_network_list)
        self.propagation_network.to(self.device)

        # placeholder for calculated user and item embeddings
        self.user_embeddings = torch.nn.init.xavier_uniform_(torch.rand((self.num_users, self.embed_k)))
        self.user_embeddings.to(self.device)
        self.item_embeddings = torch.nn.init.xavier_uniform_(torch.rand((self.num_items, self.embed_k)))
        self.item_embeddings.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


class EGCFv2Model(torch.nn.Module, ABC):
    def __init__(self,
                 num_users,
                 num_items,
                 learning_rate,
                 embed_k,
                 l_w,
                 n_layers,
                 **kwargs
                 ):
        super().__init__()

        random.seed(1234)
        np.random.seed(1234)
        torch.manual_seed(1234)
        torch.cuda.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        torch.backends.cudnn.deterministic = True

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_users = num_users
        self.num_items = num_items
        self.embed_k = embed_k
        self.learning_rate = learning_rate
        self.n_layers = n_layers

        self.Gu = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gu.to(self.device)
        self.Gi = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Gi.to(self.device)

        self.Gut = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_users, self.embed_k))))
        self.Gut.to(self.device)
        self.Git = torch.nn.Parameter(
            torch.nn.init.xavier_normal_(torch.empty((self.num_items, self.embed_k))))
        self.Git.to(self.device)

        # create node-node collaborative
        propagation_node_node_collab_list = []
        for _ in range(self.n_layers):
            propagation_node_node_collab_list.append((LGConv(), 'x, edge_index -> x'))

        self.node_node_collab_network = torch_geometric.nn.Sequential('x, edge_index',
                                                                      propagation_node_node_collab_list)
        self.node_node_collab_network.to(self.device)

        # create node-node textual
        propagation_node_node_textual_list = []
        for _ in range(self.n_layers):
            propagation_node_node_textual_list.append(
                (NodeNodeTextLayer(), 'x, edge_index -> x'))

        self.node_node_textual_network = torch_geometric.nn.Sequential('x, edge_index',
                                                                       propagation_node_node_textual_list)
        self.node_node_textual_network.to(self.device)

        # projection layer
        self.projection = torch.nn.Linear(768, self.embed_k)
        self.projection.to(self.device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


baby_egcf = {
    "num_users": 4669,
    "num_items": 5435,
    "n_layers": 4
}

boys_girls_egcf = {
    "num_users": 8806,
    "num_items": 4165,
    "n_layers": 4
}

men_egcf = {
    "num_users": 3218,
    "num_items": 7605,
    "n_layers": 2
}

baby_lightgcn = {
    "num_users": 4669,
    "num_items": 5435,
    "n_layers": 4
}

boys_girls_lightgcn = {
    "num_users": 8806,
    "num_items": 4165,
    "n_layers": 4
}

men_lightgcn = {
    "num_users": 3218,
    "num_items": 7605,
    "n_layers": 2
}

baby_dgcf = {
    "num_users": 4669,
    "num_items": 5435,
    "n_layers": 4
}

boys_girls_dgcf = {
    "num_users": 8806,
    "num_items": 4165,
    "n_layers": 4
}

men_dgcf = {
    "num_users": 3218,
    "num_items": 7605,
    "n_layers": 2
}

dataset = 'amazon_baby'
pretrained = True

train = pd.read_csv('../../../data/{0}/train_reviews.tsv'.format(dataset), sep='\t')
train = train[['USER_ID', 'ITEM_ID', 'RATING', 'REVIEW']]
public_users = {u: idx for idx, u in enumerate(set(train['USER_ID'].unique().tolist()))}
public_items = {i: idx for idx, i in enumerate(set(train['ITEM_ID'].unique().tolist()))}
train['USER_ID'] = pd.Series([public_users[u] for u in train['USER_ID'].tolist()])
train['ITEM_ID'] = pd.Series([public_items[i] for i in train['ITEM_ID'].tolist()])

if dataset == 'amazon_baby':
    egcf = EGCFv2Model(
        num_users=baby_egcf["num_users"],
        num_items=baby_egcf["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        l_w=0.0,
        n_layers=baby_egcf["n_layers"]
    )
    lightgcn = LightGCNModel(
        num_users=baby_lightgcn["num_users"],
        num_items=baby_lightgcn["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        n_layers=baby_lightgcn["n_layers"]
    )
    dgcf = DGCFModel(
        num_users=baby_dgcf["num_users"],
        num_items=baby_dgcf["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        n_layers=baby_dgcf["n_layers"]
    )

elif dataset == "amazon_boys_girls":
    egcf = EGCFv2Model(
        num_users=boys_girls_egcf["num_users"],
        num_items=boys_girls_egcf["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        l_w=0.0,
        n_layers=boys_girls_egcf["n_layers"]
    )
    lightgcn = LightGCNModel(
        num_users=boys_girls_lightgcn["num_users"],
        num_items=boys_girls_lightgcn["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        n_layers=boys_girls_lightgcn["n_layers"]
    )
    dgcf = DGCFModel(
        num_users=boys_girls_dgcf["num_users"],
        num_items=boys_girls_dgcf["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        n_layers=boys_girls_dgcf["n_layers"]
    )

else:
    egcf = EGCFv2Model(
        num_users=men_egcf["num_users"],
        num_items=men_egcf["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        l_w=0.0,
        n_layers=men_egcf["n_layers"]
    )
    lightgcn = LightGCNModel(
        num_users=boys_girls_lightgcn["num_users"],
        num_items=boys_girls_lightgcn["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        n_layers=boys_girls_lightgcn["n_layers"]
    )
    dgcf = DGCFModel(
        num_users=men_dgcf["num_users"],
        num_items=men_dgcf["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        n_layers=men_dgcf["n_layers"]
    )

if pretrained:
    checkpoint_egcf = torch.load("../../../data/{0}/egcf/model".format(dataset))
    egcf.load_state_dict(checkpoint_egcf['model_state_dict'])
    checkpoint_lightgcn = torch.load("../../../data/{0}/lightgcn/model".format(dataset))
    lightgcn.load_state_dict(checkpoint_lightgcn['model_state_dict'])
    checkpoint_dgcf = torch.load("../../../data/{0}/dgcf/model".format(dataset))
    dgcf.load_state_dict(checkpoint_dgcf['model_state_dict'])

Gut = egcf.Gut.detach().cpu().numpy()
Git = egcf.Git.detach().cpu().numpy()
Gulight = lightgcn.Gu.detach().cpu().numpy()
Gilight = lightgcn.Gi.detach().cpu().numpy()
Gudgcf = dgcf.Gu.detach().cpu().numpy()
Gidgcf = dgcf.Gi.detach().cpu().numpy()

count_items = train.groupby('ITEM_ID').size().reset_index(name='counts')
count_items = count_items.sort_values(by='counts', ascending=True)
count_items = count_items[count_items['counts'] == 5]
count_items = count_items.sample(frac=1).reset_index(drop=True)
selected_item = count_items.head(1).ITEM_ID.tolist()

item_embedding_egcf = Git[selected_item]
item_embedding_lightgcn = Gilight[selected_item]
item_embedding_dgcf = Gidgcf[selected_item]
selected_users = train[train['ITEM_ID'] == selected_item[0]]['USER_ID'].tolist()
user_embeddings_egcf = Gut[selected_users]
user_embeddings_lightgcn = Gulight[selected_users]
user_embeddings_dgcf = Gudgcf[selected_users]
X = np.concatenate((user_embeddings_egcf,
                    user_embeddings_lightgcn,
                    user_embeddings_dgcf,
                    item_embedding_egcf,
                    item_embedding_lightgcn,
                    item_embedding_dgcf), axis=0)
t_sne = TSNE(n_components=2, random_state=1234).fit_transform(X)

# egcf (reference)
x_item_egcf, y_item_egcf = t_sne[-3, 0].tolist(), t_sne[-3, 1].tolist()
plt.scatter(x_item_egcf, y_item_egcf, color='yellow', marker='*', edgecolors='black', s=200)
x_users_egcf, y_users_egcf = t_sne[:5, 0].tolist(), t_sne[:5, 1].tolist()
plt.scatter(x_users_egcf, y_users_egcf,
            color=["red", "green", "blue", "orange", "purple"])
for x, y in zip(x_users_egcf, y_users_egcf):
    plt.plot([x_item_egcf, x], [y_item_egcf, y], color='black', linewidth=1.0, ls='-')

# lightgcn
x_item_lightgcn, y_item_lightgcn = t_sne[-2, 0].tolist(), t_sne[-2, 1].tolist()
x_lightgcn_displacement = x_item_egcf - x_item_lightgcn
y_lightgcn_displacement = y_item_egcf - y_item_lightgcn
# plt.scatter(x_item_lightgcn + x_lightgcn_displacement,
#             y_item_lightgcn + y_lightgcn_displacement, color='blue', marker='*', edgecolors='black', s=200)
x_users_lightgcn, y_users_lightgcn = t_sne[5:10, 0].tolist(), t_sne[5:10, 1].tolist()
plt.scatter((np.array(x_users_lightgcn) + x_lightgcn_displacement).tolist(),
            (np.array(y_users_lightgcn) + y_lightgcn_displacement).tolist(),
            color=["red", "green", "blue", "orange", "purple"])
for x, y in zip((np.array(x_users_lightgcn) + x_lightgcn_displacement).tolist(),
                (np.array(y_users_lightgcn) + y_lightgcn_displacement).tolist()):
    plt.plot([x_item_egcf, x], [y_item_egcf, y], color='black', linewidth=1.0, ls='--')

# dgcf
x_item_dgcf, y_item_dgcf = t_sne[-1, 0].tolist(), t_sne[-1, 1].tolist()
x_dgcf_displacement = x_item_egcf - x_item_dgcf
y_dgcf_displacement = y_item_egcf - y_item_dgcf
# plt.scatter(x_item_dgcf + x_dgcf_displacement,
#             y_item_dgcf + y_dgcf_displacement, color='red', marker='*', edgecolors='black', s=200)
x_users_dgcf, y_users_dgcf = t_sne[10:15, 0].tolist(), t_sne[10:15, 1].tolist()
plt.scatter((np.array(x_users_dgcf) + x_dgcf_displacement).tolist(),
            (np.array(y_users_dgcf) + y_dgcf_displacement).tolist(),
            color=["red", "green", "blue", "orange", "purple"])
for x, y in zip((np.array(x_users_dgcf) + x_dgcf_displacement).tolist(),
                (np.array(y_users_dgcf) + y_dgcf_displacement).tolist()):
    plt.plot([x_item_egcf, x], [y_item_egcf, y], color='black', linewidth=1.0, ls=':')

# plt.legend(labels=['SimGCF', 'LightGCN', 'DGCF'], loc='best')

plt.axis('off')
plt.show()
plt.close()
