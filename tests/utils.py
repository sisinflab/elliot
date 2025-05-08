from types import SimpleNamespace
import pandas as pd


def read_dataset(dataset_path, cols=None):
    column_names = ['userId', 'itemId', 'rating', 'timestamp'] if cols is None else cols
    df_mock = pd.read_csv(dataset_path, sep='\t', names=column_names)
    return df_mock

def create_namespace(config, attr=None):
    if attr:
        config = {attr: config}
    ns = dict_to_namespace(config)
    return ns

def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
