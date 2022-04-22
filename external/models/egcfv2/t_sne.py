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


class EGCFv2Model(torch.nn.Module):
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
        self.l_w = l_w
        self.n_layers = n_layers
        self.alpha = torch.tensor([1 / (k + 1) for k in range(self.n_layers + 1)])

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

        self.softplus = torch.nn.Softplus()

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)


baby = {
    "num_users": 4669,
    "num_items": 5435,
    "n_layers": 4
}

boys_girls = {
    "num_users": 8806,
    "num_items": 4165,
    "n_layers": 4
}

men = {
    "num_users": 3218,
    "num_items": 7605,
    "n_layers": 2
}

dataset = 'amazon_baby'
pretrained = True

train = pd.read_csv('../../../data/{0}/trainingset.tsv'.format(dataset), sep='\t', header=None)
public_users = {u: idx for idx, u in enumerate(set(train[0].unique().tolist()))}
public_items = {i: idx for idx, i in enumerate(set(train[1].unique().tolist()))}
train[0] = pd.Series([public_users[u] for u in train[0].tolist()])
train[1] = pd.Series([public_items[i] for i in train[1].tolist()])

if dataset == 'amazon_baby':
    model = EGCFv2Model(
        num_users=baby["num_users"],
        num_items=baby["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        l_w=0.0,
        n_layers=baby["n_layers"]
    )
elif dataset == "amazon_boys_girls":
    model = EGCFv2Model(
        num_users=boys_girls["num_users"],
        num_items=boys_girls["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        l_w=0.0,
        n_layers=boys_girls["n_layers"]
    )
else:
    model = EGCFv2Model(
        num_users=men["num_users"],
        num_items=men["num_items"],
        learning_rate=0.0001,
        embed_k=64,
        l_w=0.0,
        n_layers=men["n_layers"]
    )

if pretrained:
    checkpoint = torch.load("../../../data/{0}/model/model".format(dataset))
    model.load_state_dict(checkpoint['model_state_dict'])
Gu = model.Gu.detach().cpu().numpy()
Gi = model.Gi.detach().cpu().numpy()
Gut = model.Gut.detach().cpu().numpy()
Git = model.Git.detach().cpu().numpy()

train.columns = ['USER_ID', 'ITEM_ID', 'RATING', 'TIME']

count_items = train.groupby('ITEM_ID').size().reset_index(name='counts')
count_items = count_items.sort_values(by='counts', ascending=True)
count_items = count_items[count_items['counts'] == 6]
count_items = count_items.sample(frac=1).reset_index(drop=True)
selected_items = count_items.head(5).ITEM_ID.tolist()

for i in selected_items:
    item_embedding = np.expand_dims(Git[i], axis=0)
    selected_users = train[train['ITEM_ID'] == i]['USER_ID'].tolist()
    user_embeddings = Gut[selected_users]
    t_sne = TSNE(n_components=2, random_state=1234).fit_transform(
        np.concatenate((user_embeddings, item_embedding), axis=0))
    x_item, y_item = t_sne[-1]
    x_users, y_users = t_sne[:-1, 0].tolist(), t_sne[:-1, 1].tolist()
    plt.scatter(x_users, y_users)
    plt.scatter(x_item, y_item, color='red')
    plt.axis('off')
    plt.show()
    plt.close()
