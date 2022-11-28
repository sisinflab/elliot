import pandas as pd

dataset = 'yelp-2018'
train_filename = 'train.txt'
test_filename = 'test.txt'

rows, cols = [], []

with open('./data/{0}/{1}'.format(dataset, train_filename), 'r') as f:
    for line in f:
        all_elements = line.split(' ')
        if '\n' not in all_elements:
            for el in all_elements[1:]:
                rows.append(int(all_elements[0]))
                cols.append(int(el))

train = pd.concat([pd.Series(rows), pd.Series(cols)], axis=1)

rows, cols = [], []

with open('./data/{0}/{1}'.format(dataset, test_filename), 'r') as f:
    for line in f:
        all_elements = line.split(' ')
        if '\n' not in all_elements:
            for el in all_elements[1:]:
                rows.append(int(all_elements[0]))
                cols.append(int(el))

test = pd.concat([pd.Series(rows), pd.Series(cols)], axis=1)

df = pd.concat([train, test], axis=0).sort_values(0).reset_index(drop=True)
df.to_csv('./data/{0}/dataset.tsv'.format(dataset), sep='\t', header=None, index=None)
