from contextlib import contextmanager
from typing import Union, Optional, Any

import numpy as np
import torch
from pytorch_toolbelt.utils import transfer_weights
from torch import nn
from torch.types import Device


def load_from_lighting(
    model: nn.Module, checkpoint_path: str, map_location: Optional[Union[int, str]] = None, strict: bool = True
) -> nn.Module:
    map_location = f"cuda:{map_location}" if type(map_location) is int else map_location
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    state_dict = {
        k.lstrip("model").lstrip("."): v for k, v in checkpoint["state_dict"].items() if k.startswith("model.")
    }

    if strict:
        model.load_state_dict(state_dict, strict=True)
    else:
        transfer_weights(model, state_dict)
    return model


def any2device(value: Any, device: Device) -> Any:
    """
    Move tensor, list of tensors, list of list of tensors,
    dict of tensors, tuple of tensors to target device.

    Args:
        value: Object to be moved
        device: target device ids

    Returns:
        Same structure as value, but all tensors moved to specified device
    """
    if isinstance(value, dict):
        return {k: any2device(v, device) for k, v in value.items()}
    elif isinstance(value, (tuple, list)):
        return [any2device(v, device) for v in value]
    elif torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    elif isinstance(value, (np.ndarray, np.void)) and value.dtype.fields is not None:
        return {k: any2device(value[k], device) for k in value.dtype.fields.keys()}
    return value


@contextmanager
def evaluating(net: nn.Module):
    """Temporarily switch to evaluation mode."""
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()
