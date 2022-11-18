import os
import pandas as pd
import argparse
import numpy as np
import random
import networkx
import torch
from networkx.algorithms import bipartite
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils.dropout import dropout_node, dropout_edge, dropout_path


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph sampling (Node Dropout, Edge Dropout, Random Walking).")
    parser.add_argument('--dataset', nargs='?', default='last-fm', help='dataset name')
    parser.add_argument('--filename', nargs='?', default='dataset.txt', help='filename')
    parser.add_argument('--sampling', nargs='+', type=str, default=['ND', 'ED', 'RW'],
                        help='graph sampling strategy')
    parser.add_argument('--number_dropout_ratios', nargs='?', type=int, default=500,
                        help='dropout ratios')
    parser.add_argument('--num_layers', nargs='?', type=int, default=4,
                        help='number of layers (only for RW)')
    parser.add_argument('--num_random_users', nargs='?', type=int, default=100,
                        help='number of random users for statistics')
    parser.add_argument('--num_random_items', nargs='?', type=int, default=100,
                        help='number of random items for statistics')
    parser.add_argument('--random_seed', nargs='?', type=int, default=42,
                        help='random seed for reproducibility')

    return parser.parse_args()


args = parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
torch.cuda.manual_seed(args.random_seed)
torch.cuda.manual_seed_all(args.random_seed)
torch.backends.cudnn.deterministic = True


def calculate_statistics(data, info):
    # basic metrics
    users = torch.unique(data[0]).shape[0]
    items = torch.unique(data[1]).shape[0]
    m = data.shape[1]
    delta_G = m / (users * items)
    k_users = m / users
    k_items = m / items

    # rescale nodes indices to feed the edge_index into networkx
    current_public_to_private_users = {u.item(): idx for idx, u in enumerate(torch.unique(data[0]))}
    current_public_to_private_items = {i.item(): idx + users for idx, i in enumerate(torch.unique(data[1]))}
    rescaled_rows = [current_public_to_private_users[u.item()] for u in data[0]]
    rescaled_cols = [current_public_to_private_items[i.item()] for i in data[1]]
    rescaled_edge_index = np.array([rescaled_rows, rescaled_cols])
    rescaled_edge_index = torch.tensor(rescaled_edge_index, dtype=torch.int64)
    torch_geometric_dataset = Data(edge_index=rescaled_edge_index, num_nodes=users + items)
    networkx_dataset = to_networkx(torch_geometric_dataset, to_undirected=True)

    # calculate average eccentricity
    random_users = random.sample(list(range(users)), args.num_random_users)
    random_items = random.sample(list(range(users, users + items)), args.num_random_users)
    eccentricity_users = list(dict(networkx.eccentricity(networkx_dataset, v=random_users)).values())
    eccentricity_items = list(dict(networkx.eccentricity(networkx_dataset, v=random_items)).values())
    average_eccentricity_users = sum(eccentricity_users) / args.num_random_users
    average_eccentricity_items = sum(eccentricity_items) / args.num_random_users

    # calculate clustering coefficients
    average_clustering_dot_users = bipartite.average_clustering(networkx_dataset, nodes=random_users, mode='dot')
    average_clustering_dot_items = bipartite.average_clustering(networkx_dataset, nodes=random_items, mode='dot')
    average_clustering_min_users = bipartite.average_clustering(networkx_dataset, nodes=random_users, mode='min')
    average_clustering_min_items = bipartite.average_clustering(networkx_dataset, nodes=random_items, mode='min')
    average_clustering_max_users = bipartite.average_clustering(networkx_dataset, nodes=random_users, mode='max')
    average_clustering_max_items = bipartite.average_clustering(networkx_dataset, nodes=random_items, mode='max')

    # calculate node redundancy
    node_redundancy_users = list(dict(networkx.node_redundancy(networkx_dataset, v=random_users)).values())
    node_redundancy_items = list(dict(networkx.node_redundancy(networkx_dataset, v=random_items)).values())
    average_node_redundancy_users = sum(node_redundancy_users) / args.num_random_users
    average_node_redundancy_items = sum(node_redundancy_items) / args.num_random_users

    stats_dict = {
        'users': users,
        'items': items,
        'interactions': m,
        'delta_G': delta_G,
        'k_users': k_users,
        'k_items': k_items,
        'eccentricity_users': average_eccentricity_users,
        'eccentricity_items': average_eccentricity_items,
        'clustering_dot_users': average_clustering_dot_users,
        'clustering_dot_items': average_clustering_dot_items,
        'clustering_min_users': average_clustering_min_users,
        'clustering_min_items': average_clustering_min_items,
        'clustering_max_users': average_clustering_max_users,
        'clustering_max_items': average_clustering_max_items,
        'node_redundancy_users': average_node_redundancy_users,
        'node_redundancy_items': average_node_redundancy_items
    }

    return info.update(stats_dict)


def graph_sampling():
    # load public dataset
    dataset = pd.read_csv(f'./data/{args.dataset}/{args.filename}', sep='\t', header=None)

    # calculate initial statistics
    num_users = dataset[0].nunique()
    num_items = dataset[1].nunique()
    num_interactions = len(dataset)  # graph is undirected
    print('\n\nSTART GRAPH SAMPLING...')
    print(f'DATASET: {args.dataset}')
    print(f'Number of users: {num_users}')
    print(f'Number of items: {num_items}')
    print(f'Number of interactions: {num_interactions}')
    print(f'Density: {num_interactions / (num_users * num_items)}')

    # calculate dropout ratios
    dropout_ratios = [np.random.uniform(0, 1) for _ in range(args.number_dropout_ratios)]

    # create public-private/private-public dictionaries
    public_to_private_users = {u: idx for idx, u in enumerate(dataset[0].unique().tolist())}
    public_to_private_items = {i: idx + num_users for idx, i in enumerate(dataset[1].unique().tolist())}
    private_to_public_users = {idx: u for u, idx in public_to_private_users.items()}
    private_to_public_items = {idx: i for i, idx in public_to_private_items.items()}

    # get rows and cols for the dataset and convert them into private
    rows = [public_to_private_users[r] for r in dataset[0].tolist()]
    cols = [public_to_private_items[c] for c in dataset[1].tolist()]

    # create edge index
    edge_index = np.array([rows, cols])
    edge_index = torch.tensor(edge_index, dtype=torch.int64)

    # create dictionary to store
    dictionary = {gss: [] for gss in args.sampling}
    filename_no_extension = args.filename.split('.')[0]
    extension = args.filename.split('.')[1]

    for gss in args.sampling:
        if gss == 'ND':
            if not os.path.exists(f'./data/{args.dataset}/node-dropout/'):
                os.makedirs(f'./data/{args.dataset}/node-dropout/')
            for dr in dropout_ratios:
                print(f'\n\nRunning NODE DROPOUT with dropout ratio: {dr}')
                sampled_edge_index, _, _ = dropout_node(edge_index, p=dr, num_nodes=num_users + num_items)
                dictionary['ND'].append(calculate_statistics(sampled_edge_index, info={'strategy': 'node dropout',
                                                                                       'dropout': dr}))
                sampled_rows = [private_to_public_users[r] for r in sampled_edge_index[0].tolist()]
                sampled_cols = [private_to_public_items[c] for c in sampled_edge_index[1].tolist()]
                sampled_dataset = pd.concat([pd.Series(sampled_rows), pd.Series(sampled_cols)], axis=1)
                sampled_dataset.to_csv(
                    f'./data/{args.dataset}/node-dropout/{filename_no_extension}-{dr}-{args.num_random_users}-{args.num_random_items}.{extension}',
                    sep='\t', header=None, index=None)
        elif gss == 'ED':
            if not os.path.exists(f'./data/{args.dataset}/edge-dropout/'):
                os.makedirs(f'./data/{args.dataset}/edge-dropout/')
            for dr in args.dropout_ratio:
                print(f'\n\nRunning EDGE DROPOUT with dropout ratio: {dr}')
                n_interactions_to_drop = round(num_interactions * dr)
                sampled_dataset = dataset.sample(n=num_interactions - n_interactions_to_drop,
                                                 random_state=args.random_seed).reset_index(drop=True)
                sampled_dataset.to_csv(f'./data/{args.dataset}/{filename_no_extension}-edge-dropout-{dr}.{extension}',
                                       sep='\t', header=None, index=None)
        elif gss == 'RW':
            if not os.path.exists(f'./data/{args.dataset}/random-walk/'):
                os.makedirs(f'./data/{args.dataset}/random-walk/')
            for dr in args.dropout_ratio:
                for nl in range(args.num_layers):
                    print(f'\n\nRunning RANDOM WALK with dropout ratio: {dr} at layer: {nl + 1}')
                    n_interactions_to_drop = round(num_interactions * dr)
                    sampled_dataset = dataset.sample(n=num_interactions - n_interactions_to_drop,
                                                     random_state=args.random_seed).reset_index(drop=True)
                    sampled_dataset.to_csv(
                        f'./data/{args.dataset}/{filename_no_extension}-random-walk-{dr}-{nl + 1}.{extension}',
                        sep='\t', header=None, index=None)
        else:
            raise NotImplementedError('This graph sampling strategy has not been implemented yet!')
    print('\n\nEND GRAPH SAMPLING...')


if __name__ == '__main__':
    graph_sampling()
