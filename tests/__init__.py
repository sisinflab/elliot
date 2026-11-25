from itertools import product
# import functools
# import time

from elliot.utils.folder import parent_dir


test_path = parent_dir(__file__)


# def time_single_test(func):
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):
#         start = time.perf_counter()
#         try:
#             return func(*args, **kwargs)
#         finally:
#             end = time.perf_counter()
#             duration = end - start
#             print(f"[{func.__name__}] executed in {duration:.4f} seconds")
#     return wrapper


def generate_param_combinations(key_list, values, base=None):
    if base is None:
        base_list = [{}] * len(key_list)
    elif not isinstance(base, list):
        base_list = [base] * len(key_list)
    else:
        base_list = base
    result = []
    for keys, base in zip(key_list, base_list):
        if not isinstance(keys, tuple):
            keys = (keys,)
        value_lists = [values[k] for k in keys]
        for combo in product(*value_lists):
            overrides = dict(zip(keys, combo))
            config = {**base, **overrides}
            result.append(config)
    return result
