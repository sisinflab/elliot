def import_model_by_backend(tensorflow_cmd, pytorch_cmd):
    import sys
    for _backend in sys.modules["external"].backend:
        if _backend == "tensorflow":
            exec(tensorflow_cmd)
        elif _backend == "pytorch":
            exec(pytorch_cmd)
            break


from .most_popular import MostPop
from .msap import MSAPMF
from .AdversarialMF import AdversarialMF

import sys
for _backend in sys.modules["external"].backend:
    if _backend == "tensorflow":
        pass
    elif _backend == "pytorch":
        from .ngcf import NGCF
        from .lightgcn import LightGCN
        from .pinsage import PinSage
        from .gat import GAT
        from .gcmc import GCMC
        from .disen_gcn import DisenGCN
        from .mmgcn import MMGCN
        from .dgcf import DGCF
        from .egcf import EGCF

