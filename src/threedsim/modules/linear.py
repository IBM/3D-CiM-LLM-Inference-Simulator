import math

import torch
from torch import fx

from ..accelerator import Tier
from ..utils import digital_add, get_logger
from .base import BaseModule

fx.wrap(digital_add)


@fx.wrap
def grouped_linear(token: torch.Tensor, weight: torch.Tensor, op_info: dict = None):
    return (token @ weight).flatten()


class Linear(BaseModule, torch.nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = False,
        device="meta",
        dtype=None,
        name=None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert not bias, "bias not supported"
        self.logger = get_logger("Linear")
        self.mapping: list[list[tuple[int]]] = None
        self.in_features = in_features
        self.out_features = out_features
        self.name: str = name
        self.device: str = device
        self.dtype = dtype
        self.use_linear: bool = False

        self.weight = torch.nn.Parameter(
            torch.randn((in_features, out_features), device=device)
        )

    def forward(self, token: torch.Tensor, op_info: dict = None):
        assert (
            hasattr(self, "mapping") and self.mapping is not None
        ), f"Forgot to map {self.name}"

        if not isinstance(op_info, dict):
            op_info = {"token_id": 0, "seq_len": 1, "layer_name": self.name}
        else:
            op_info = {**op_info, "layer_name": self.name}

        # mvm is applied as x @ W
        if self.use_linear:
            op_info = {
                **op_info,
                "mapping": self.mapping,
                "shape": (self.in_features, self.out_features),
            }
            return grouped_linear(token, self.weight, op_info=op_info)

        tier_n_rows, tier_n_cols = self.accelerator.config.tier_shape
        assert (
            self.traceable or token.ndim == 1
        ), "Token must have one dimension to avoid ambiguity"
        assert self.traceable or math.ceil(token.numel() / tier_n_rows) == len(
            self.mapping
        ), "illegal mapping"
        # let W be the matrix that we multiply the token with
        # the token is split into y chunks and called c_token. c_token[0] is multiplied with W[0,0], W[0,1], ...
        c_token = torch.split(token, split_size_or_sections=tier_n_rows)
        outputs = []
        for i in range(self.n_vertical):
            hor_output = []
            for j in range(self.n_horizontal):
                tile_idx, tier_idx, utilization, n_rows, n_cols = self.mapping[i][j]
                # we want to save some more information in this op
                linear_op_info = {
                    "tile_idx": tile_idx,
                    "tier_idx": tier_idx,
                    "utilization": utilization,
                    "n_rows": n_rows,
                    "n_cols": n_cols,
                    **op_info,
                }
                hor_output.append(
                    self.accelerator.tiles[tile_idx].tiers[tier_idx](
                        c_token[i], op_info=linear_op_info
                    )
                )
            outputs.append(hor_output)

        # Add all results in the Y axis if self.n_vertical > 1
        if self.n_vertical > 1:
            add_op_info = {"size": self.accelerator.config.tier_shape[1], **op_info}
            aggregated_output = []
            for j in range(self.n_horizontal):
                temp_res = digital_add(
                    outputs[0][j], outputs[1][j], op_info=add_op_info
                )
                for i in range(2, self.n_vertical):
                    temp_res = digital_add(temp_res, outputs[i][j], op_info=add_op_info)
                aggregated_output.append(temp_res)
        else:
            aggregated_output = outputs[0]

        # Concatenate all results in the X axis if self.n_horizontal > 1
        output = (
            torch.cat(aggregated_output)
            if self.n_horizontal > 1
            else aggregated_output[0]
        )

        return output.flatten()[: self.out_features]

    def set_name(self, name):
        assert (
            hasattr(self, "mapping") and self.mapping is not None
        ), f"Forgot to map {name}"
        self.name = name
        for i in range(self.n_vertical):
            for j in range(self.n_horizontal):
                tile_idx, tier_idx, utilization, n_rows, n_cols = self.mapping[i][j]
                t: Tier = self.accelerator.tiles[tile_idx].tiers[tier_idx]
                t.set_name(f"{self.name}_{t.name}")

    def set_mapping(self, mapping: list[list[tuple[int]]]):
        assert self.accelerator is not None, "Need to call assign_acc(model, acc)"
        self.mapping = mapping
        self.n_vertical = len(mapping)
        self.n_horizontal = len(mapping[0])
        for sl in mapping:
            for t in sl:
                tile_idx, tier_idx, utilization, n_rows, n_cols = t
                assert (
                    not self.accelerator.tiles[tile_idx].tiers[tier_idx].is_mapped
                ), f"tile {tile_idx} tier {tier_idx} already mapped"
                self.accelerator.tiles[tile_idx].tiers[tier_idx].is_mapped = True
