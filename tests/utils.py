import functools
import time
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

test_path = Path(__file__).parent / 'data'
data_path = Path(__file__).parent.parent / 'data'


def time_single_test(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            end = time.perf_counter()
            duration = end - start
            print(f"[{func.__name__}] executed in {duration:.4f} seconds")
    return wrapper

def read_dataset(dataset_path, cols=None):
    column_names = ['userId', 'itemId', 'rating', 'timestamp'] if cols is None else cols
    df = pd.read_csv(dataset_path, sep='\t', names=column_names)
    df_mock = df.dropna(axis=1, how='all')
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
