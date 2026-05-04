from argparse import Namespace
from typing import Tuple
from .factory import MoELoRAOptions

def namespace_to_moe_opt(ns: Namespace) -> Tuple[MoELoRAOptions, Namespace]:
    """Filter Namespace to create MoELoRAOptions and return the unused fields."""
    valid_keys = MoELoRAOptions.__dataclass_fields__.keys()
    filtered_dict = {k: v for k, v in vars(ns).items() if k in valid_keys}
    discarded_dict = {k: v for k, v in vars(ns).items() if k not in valid_keys}
    return MoELoRAOptions(**filtered_dict), Namespace(**discarded_dict)