from typing import Tuple, List, Dict
from torch.nn.utils import prune

def l1_prune(features, amount) -> Tuple:
    prune_params = tuple(features)
    prune.global_unstructured(
        parameters=prune_params,
        pruning_method=prune.L1Unstructured,
        amount=amount
    )
    for module, name in prune_params:
        prune.remove(module, name)
    return prune_params

def l2_prune(features, amount) -> Tuple:
    assert amount >= 0
    prune_params = tuple(features)
    for module, name in prune_params:
        prune.ln_structured(module, name, amount=amount, n=2, dim=0)
        prune.remove(module, name)
    return prune_params