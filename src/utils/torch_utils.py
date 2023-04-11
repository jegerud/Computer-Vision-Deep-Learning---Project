import random
import numpy as np
import torch

AMP_enabled = False

def set_AMP(value: bool):
    global AMP_enabled
    AMP_enabled = value


def AMP():
    return AMP_enabled


def _to_cuda(element):
    return element.to(get_device(), non_blocking=True)

def to_cuda(elements):
    if isinstance(elements, tuple) or isinstance(elements, list):
        return [_to_cuda(x) for x in elements]
    if isinstance(elements, dict):
        return {k: _to_cuda(v) for k,v in elements.items()}
    return _to_cuda(elements)

def move_to(obj):
    device = get_device()
    # print(f"Object type: {type(obj)}")
    # Recursively move the elements of an object to device, used for annotations
    if torch.is_tensor(obj): 
        return obj.to(device)
    elif isinstance(obj, dict):
        res = {}
        for k, v in obj.items(): res[k] = move_to(v)
        return res
    elif isinstance(obj, list):
        res = []
        for v in obj: res.append(move_to(v))
        return res
    else:
        print(f"Type obj: {type(obj)}")
        raise TypeError("Invalid type for move_to")


def get_device() -> torch.device:
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
