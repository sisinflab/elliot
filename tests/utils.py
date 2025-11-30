import functools
import time
from pathlib import Path
from types import SimpleNamespace

from elliot.utils.read import read_tabular

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


def read_dataset(dataset_path, custom_cols=None, custom_dtypes=None, header=False):
    cols = ['userId', 'itemId', 'rating', 'timestamp']
    datatypes = ['str', 'str', 'float', 'float']

    selected_cols = custom_cols if custom_cols is not None else cols
    selected_dtypes = custom_dtypes if custom_dtypes is not None else datatypes

    df = read_tabular(
        dataset_path,
        cols=selected_cols,
        datatypes=selected_dtypes,
        sep='\t',
        header=header
    )
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
