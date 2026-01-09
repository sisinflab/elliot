import functools
import time
from pathlib import Path

from elliot.namespace import NameSpaceModel

test_path = Path(__file__).parent
data_path = str(test_path / "data" / "{0}")


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


def create_namespace(config, source_path):
    model = NameSpaceModel(config, str(test_path), str(source_path))
    model.fill_base()
    return model.base_namespace
