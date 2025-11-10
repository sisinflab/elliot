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


def xavier_uniform_init(module):
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            zeros_(module.bias.data)
