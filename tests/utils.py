import functools
import time
import pandas as pd
from pathlib import Path
from types import SimpleNamespace

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
    default_cols = ['userId', 'itemId', 'rating', 'timestamp']
    df_preview = pd.read_csv(dataset_path, sep='\t', nrows=1, header=None)

    column_names = cols if (df_preview.shape[1] < 4 and cols is not None) else default_cols

    df = pd.read_csv(dataset_path, sep='\t', names=column_names, header=None)
    df_clean = df.dropna(axis=1, how='all')

    return df_clean


def create_namespace(config, attr=None):
    if attr:
        config = {attr: config}
    ns = _dict_to_namespace(config)
    return ns

def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    else:
        return d
