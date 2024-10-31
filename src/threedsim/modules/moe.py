from typing import Optional

import torch
from torch import fx

from ..utils import add_dependency, digital_add, get_logger
from .activation import get_activation_fn
from .base import BaseModule
from .linear import Linear

fx.wrap(digital_add)
fx.wrap(add_dependency)


def _uniform_density(k: int, num_experts: int, device):
    assert k in [1, 2] and isinstance(k, int), "k must be int and in [1,2]"
    if k == 1:
        probability_density = torch.rand(num_experts, device=device)
    else:
        probability_density = torch.rand(num_experts, num_experts, device=device)
        t_i, t_j = torch.triu_indices(num_experts, num_experts)
        probability_density.T[t_i, t_j] = probability_density[t_i, t_j]
        probability_density.fill_diagonal_(0.0)
    density = probability_density / probability_density.sum()
    return density


def _sample(density, num_experts, k):
    if density is None:
        return torch.randperm(num_experts)[:k].tolist()
    if density.ndim == 1:
        return [int(torch.multinomial(density.flatten(), num_samples=1))]
    if density.ndim == 2:
        num_experts = density.size(0)
        top_ks = int(torch.multinomial(density.flatten(), num_samples=1))
        top_ks = [top_ks // num_experts, top_ks % num_experts]
        return top_ks


class Expert(BaseModule, torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        activation: str,
        device="meta",
        dtype=None,
    ):
        super().__init__()
        self.dim_feedforward = dim_feedforward
        self.ffn1 = Linear(d_model, dim_feedforward, device=device, dtype=dtype)
        self.ffn2 = Linear(dim_feedforward, d_model, device=device, dtype=dtype)
        self.activation = get_activation_fn(activation)

    def forward(self, input: torch.Tensor, op_info: dict = None):
        x = self.ffn1(input, op_info=op_info)
        x = self.activation(x, op_info={"size": self.dim_feedforward, **op_info})
        x = self.ffn2(x, op_info=op_info)
        return x


class MoELayer(BaseModule, torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int,
        num_experts: int,
        k: int,
        activation: str = "relu",
        density: Optional[torch.Tensor] = None,
        device="meta",
        dtype=None,
        name=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.k = k
        self.activation = activation
        self.density = density
        if density is None and k in [1, 2]:
            self.density = _uniform_density(k, num_experts=num_experts, device="cpu")
        if k > 2:
            assert density is None, "Density must be None for k > 2"
        self.logger = get_logger("MoELayer")
        self.name: str = name
        self.device: str = device
        self.dtype = dtype

        # moe related
        self.router = Linear(d_model, num_experts, device=device)
        self.experts = torch.nn.ModuleList(
            [
                Expert(d_model, dim_feedforward, activation, device, dtype)
                for _ in range(num_experts)
            ]
        )

    def forward(self, token: torch.Tensor, op_info: dict = None):
        # first, pass through the router
        router_logits = self.router(token, op_info=op_info)
        sm_logits = torch.nn.functional.softmax(router_logits, dim=0)
        # top-k also returns indices, but we get the indices from
        # a density that we can set in the layer
        values, _ = sm_logits.topk(k=self.k)
        indices = _sample(density=self.density, num_experts=self.num_experts, k=self.k)
        values, token = add_dependency(values, token)
        # indices, values = add_dependency(indices, values)
        output = values[0] * self.experts[indices[0]](token, op_info=op_info)
        for idx in range(1, self.k):
            expert_idx, scale = indices[idx], values[idx]
            expert_out = scale * self.experts[expert_idx](token, op_info=op_info)
            output = digital_add(
                output, expert_out, op_info={"size": self.dim_feedforward, **op_info}
            )
        return output
