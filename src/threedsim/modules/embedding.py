import torch
import torch.fx as fx

from ..accelerator import Tier
from ..graph.processing import get_op_info, seq_id_from_op
from .base import BaseModule


class Embedding(BaseModule, torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device="meta",
        dtype=None,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = torch.nn.Parameter(
            torch.randn((num_embeddings, embedding_dim), device=device)
        )

    def forward(self, token: torch.Tensor, op_info: dict = None) -> torch.Tensor:
        assert (
            hasattr(self, "mapping") and self.mapping is not None
        ), f"Forgot to map {self.name}"
        assert (
            self.traceable or isinstance(token, int) or token.ndim == 0
        ), "Token must be a single tensor number"
        if not isinstance(op_info, dict):
            op_info = {"token_id": 0, "seq_len": 1}

        if not self.traceable:
            row_tier, row_crossbar = self._get_tier_and_row(token)
            encode_vector = self._gen_encode_vector(row_crossbar)
        else:
            # To allow static tracing, generate a static vector
            # It will change afterwards by calling the
            # `assign_embedding_tiers_to_traced_graph` function
            row_tier = 0
            encode_vector = _gen_encode_vector_for_trace(
                token, self.accelerator.config.tier_shape[0]
            )

        outputs = []
        for j in range(self.n_horizontal):
            tile_idx, tier_idx, utilization, n_rows, n_cols = self.mapping[row_tier][j]
            # we want to save some more information in this op
            embedding_op_info = {
                "layer_name": self.name,
                "tile_idx": tile_idx,
                "tier_idx": tier_idx,
                "utilization": utilization,
                "n_rows": n_rows,
                "n_cols": n_cols,
                "type": f"embedding_{j}",
                **op_info,
            }
            outputs.append(
                self.accelerator.tiles[tile_idx].tiers[tier_idx](
                    encode_vector, op_info=embedding_op_info
                )
            )

        return torch.cat([output for output in outputs])[: self.embedding_dim]

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

    def _get_tier_and_row(self, token: torch.Tensor) -> tuple[int, int]:
        if isinstance(token, torch.Tensor):
            token = token.item()
        tier_n_rows, _ = self.accelerator.config.tier_shape
        row_tier = token // tier_n_rows
        row_crossbar = token % tier_n_rows
        return row_tier, row_crossbar

    def _gen_encode_vector(self, row_crossbar: int, one_hot=True):
        tier_n_rows, _ = self.accelerator.config.tier_shape
        if one_hot:
            encode_vector = torch.zeros(tier_n_rows, **self.factory_kwargs)
            encode_vector[row_crossbar] = 1
        else:
            raise NotImplementedError("Only one-hot encoding supported")
        return encode_vector


@fx.wrap
def _gen_encode_vector_for_trace(token, pad_size):
    return torch.zeros(pad_size)


def assign_embedding_tiers_to_traced_graph(
    graph: fx.Graph, tokens: torch.Tensor, embedding_layers_dict: dict[str, Embedding]
):
    for op in graph.nodes:
        if "tier_linear" in op.name:
            info_dict = get_op_info(op)
            if info_dict.get("type") and "embedding" in info_dict.get("type"):
                seq_id = seq_id_from_op(op)
                token_id = info_dict["token_id"]
                embedding_layer = embedding_layers_dict[info_dict["layer_name"]]
                token_value = tokens[seq_id][token_id]
                tier, _ = embedding_layer._get_tier_and_row(token_value)
                horizontal_idx = int(info_dict["type"].split("_")[1])
                tile_idx, tier_idx = embedding_layer.mapping[tier][horizontal_idx]
                new_kwargs = dict(op.kwargs)
                new_kwargs["op_info"] = dict(new_kwargs["op_info"])
                new_kwargs["op_info"]["tile_idx"] = tile_idx
                new_kwargs["op_info"]["tier_idx"] = tier_idx
                op.kwargs = new_kwargs
