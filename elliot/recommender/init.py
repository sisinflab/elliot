import numpy as np
from torch import nn
from torch.nn.init import xavier_normal_, xavier_uniform_, zeros_, normal_


def zeros_init(module):
    if isinstance(module, nn.Embedding):
        zeros_(module.weight.data)
    elif isinstance(module, np.ndarray):
        module.fill(0.0)


def normal_init(module, mean=0.0, std=0.1):
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data, mean=mean, std=std)
    elif isinstance(module, nn.Linear):
        normal_(module.weight.data, mean=mean, std=std)
        if module.bias is not None:
            zeros_(module.bias.data)
    elif isinstance(module, nn.Conv2d):
        normal_(module.weight.data, mean=mean, std=std)
        if module.bias is not None:
            zeros_(module.bias.data)
    elif isinstance(module, np.ndarray):
        module[:] = np.random.normal(loc=mean, scale=std, size=module.shape)


def xavier_normal_init(module):
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            zeros_(module.bias.data)
    elif isinstance(module, np.ndarray):
        module[:] = xavier_init(module.shape, init=np.random.normal)


def xavier_uniform_init(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            zeros_(module.bias.data)
    elif isinstance(module, np.ndarray):
        module[:] = xavier_init(module.shape, init=np.random.uniform)

def xavier_init(shape, init=np.random.normal, fan_in=None, fan_out=None, gain=1.0, dtype=np.float32):
    if fan_in is None or fan_out is None:
        if len(shape) < 2:
            raise ValueError("Serve fan_in/fan_out oppure una shape >=2")
        fan_out = shape[0]
        fan_in = shape[1]
    limit = gain * np.sqrt(6.0 / (fan_in + fan_out))
    return init(-limit, limit, size=shape).astype(dtype)