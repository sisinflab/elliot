import functools
import time

from elliot.namespace import NameSpaceModel
from elliot.utils.folder import parent_dir, path_joiner

test_path = parent_dir(__file__)
data_folder = path_joiner(test_path, "data", "{0}")
dataset_path = path_joiner(data_folder, "dataset.tsv")


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
    ns_model = NameSpaceModel(config, str(test_path), str(source_path))
    ns_model.fill_base()
    return ns_model
