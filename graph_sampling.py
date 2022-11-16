import pandas as pd
import argparse
import numpy as np
import random


def parse_args():
    parser = argparse.ArgumentParser(description="Run graph sampling (Node Dropout, Edge Dropout, Random Walking).")
    parser.add_argument('--dataset', nargs='?', default='last-fm', help='dataset name')
    parser.add_argument('--filename', nargs='?', default='dataset.txt', help='filename')
    parser.add_argument('--sampling', nargs='+', type=str, default=['ND', 'ED', 'RW'],
                        help='graph sampling strategy')
    parser.add_argument('--dropout_ratio', nargs='+', type=float, default=[0.1, 0.3, 0.5, 0.7, 0.9],
                        help='dropout ratios')
    parser.add_argument('--num_layers', nargs='?', type=int, default=3,
                        help='number of layers (only for RW)')
    parser.add_argument('--random_seed', nargs='?', type=int, default=42,
                        help='random seed for reproducibility')

    return parser.parse_args()


args = parse_args()

random.seed(args.random_seed)
np.random.seed(args.random_seed)


def graph_sampling():
    dataset = pd.read_csv(f'./data/{args.dataset}/{args.filename}', sep='\t', header=None)
    filename_no_extension = args.filename.split('.')[0]
    extension = args.filename.split('.')[1]
    num_users = dataset[0].nunique()
    num_items = dataset[1].nunique()
    users = dataset[0].unique().tolist()
    items = dataset[1].unique().tolist()
    num_interactions = len(dataset)
    print('\n\nSTART GRAPH SAMPLING...')
    with open(f'./data/{args.dataset}/statistics.txt', 'w') as f:
        print(f'DATASET: {args.dataset}', file=f)
        print(f'Number of users: {num_users}', file=f)
        print(f'Number of items: {num_items}', file=f)
        print(f'Number of interactions: {num_interactions}', file=f)
        print(f'Sparsity: {1 - (num_interactions / (num_users * num_items))}', file=f)
        for gss in args.sampling:
            if gss == 'ND':
                for dr in args.dropout_ratio:
                    print(f'\n\nRunning NODE DROPOUT with dropout ratio: {dr}', file=f)
                    n_users_to_drop = round(num_users * dr)
                    n_items_to_drop = round(num_items * dr)
                    users_to_drop = random.sample(users, n_users_to_drop)
                    items_to_drop = random.sample(items, n_items_to_drop)
                    sampled_dataset = dataset[
                        ~((dataset[0].isin(users_to_drop)) | (dataset[1].isin(items_to_drop)))].reset_index(drop=True)
                    print(f'Number of users: {sampled_dataset[0].nunique()}', file=f)
                    print(f'Number of items: {sampled_dataset[1].nunique()}', file=f)
                    print(f'Number of interactions: {len(sampled_dataset)}', file=f)
                    print(f'Sparsity: {1 - (len(sampled_dataset) / (sampled_dataset[0].nunique() * sampled_dataset[1].nunique()))}', file=f)
                    sampled_dataset.to_csv(f'./data/{args.dataset}/{filename_no_extension}-node-dropout-{dr}.{extension}',
                                           sep='\t', header=None, index=None)
            elif gss == 'ED':
                for dr in args.dropout_ratio:
                    print(f'\n\nRunning EDGE DROPOUT with dropout ratio: {dr}', file=f)
                    n_interactions_to_drop = round(num_interactions * dr)
                    sampled_dataset = dataset.sample(n=num_interactions - n_interactions_to_drop,
                                                     random_state=args.random_seed).reset_index(drop=True)
                    print(f'Number of users: {sampled_dataset[0].nunique()}', file=f)
                    print(f'Number of items: {sampled_dataset[1].nunique()}', file=f)
                    print(f'Number of interactions: {len(sampled_dataset)}', file=f)
                    print(f'Sparsity: {1 - (len(sampled_dataset) / (sampled_dataset[0].nunique() * sampled_dataset[1].nunique()))}', file=f)
                    sampled_dataset.to_csv(f'./data/{args.dataset}/{filename_no_extension}-edge-dropout-{dr}.{extension}',
                                           sep='\t', header=None, index=None)
            elif gss == 'RW':
                for dr in args.dropout_ratio:
                    for nl in range(args.num_layers):
                        print(f'\n\nRunning RANDOM WALK with dropout ratio: {dr} at layer: {nl + 1}', file=f)
                        n_interactions_to_drop = round(num_interactions * dr)
                        sampled_dataset = dataset.sample(n=num_interactions - n_interactions_to_drop,
                                                         random_state=args.random_seed).reset_index(drop=True)
                        print(f'Number of users: {sampled_dataset[0].nunique()}', file=f)
                        print(f'Number of items: {sampled_dataset[1].nunique()}', file=f)
                        print(f'Number of interactions: {len(sampled_dataset)}', file=f)
                        print(f'Sparsity: {1 - (len(sampled_dataset) / (sampled_dataset[0].nunique() * sampled_dataset[1].nunique()))}', file=f)
                        sampled_dataset.to_csv(
                            f'./data/{args.dataset}/{filename_no_extension}-random-walk-{dr}-{nl + 1}.{extension}',
                            sep='\t', header=None, index=None)
            else:
                raise NotImplementedError('This graph sampling strategy has not been implemented yet!')
        print('\n\nEND GRAPH SAMPLING...')


if __name__ == '__main__':
    graph_sampling()
