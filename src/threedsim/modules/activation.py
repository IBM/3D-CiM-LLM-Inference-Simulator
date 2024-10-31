from typing import Callable

import torch
from torch import Tensor


@torch.fx.wrap
def digital_relu(input, inplace, op_info: dict = None):
    return torch.nn.functional.relu(input, inplace=inplace)


@torch.fx.wrap
def digital_gelu(input, approximate, op_info: dict = None):
    return torch.nn.functional.gelu(input, approximate=approximate)


class ReLU(torch.nn.ReLU):
    def __init__(self, inplace: bool = False):
        super().__init__(inplace)

    def forward(self, input: Tensor, op_info: dict = None) -> Tensor:
        return digital_relu(input, self.inplace, op_info=op_info)


class GELU(torch.nn.GELU):
    def __init__(self, approximate: str = "none") -> None:
        super().__init__(approximate)

    def forward(self, input: Tensor, op_info: dict = None) -> Tensor:
        return digital_gelu(input, self.approximate, op_info=op_info)


def get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    """
    Return object of layer corresponding to activation function.

    Args:
        activation (str): "relu" or "gelu"

    Returns:
        Callable[[Tensor], Tensor]: Activation function.
    """
    if activation == "relu":
        return ReLU()
    elif activation == "gelu":
        return GELU()
