import torch
from torch import Tensor
from torch.nn.modules.normalization import _shape_t

from .base import BaseModule


@torch.fx.wrap
def layer_norm(input, normalized_shape, weight, bias, eps, op_info: dict = None):
    return torch.nn.functional.layer_norm(input, normalized_shape, weight, bias, eps)


class LayerNorm(BaseModule, torch.nn.LayerNorm):
    def __init__(
        self,
        normalized_shape: _shape_t,
        eps: float = 0.00001,
        elementwise_affine: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            normalized_shape, eps, elementwise_affine, device=device, dtype=dtype
        )

    def forward(self, input: Tensor, op_info: dict = None) -> Tensor:
        return layer_norm(
            input,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
            op_info=op_info,
        )
