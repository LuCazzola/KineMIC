from torch import nn

from .loratorch.modules.layers import LoRALayer

def has_adapter(model: nn.Module) -> bool:
    instances = (LoRALayer,)
    for m in model.modules():
        if isinstance(m, instances):
            return True
    return False