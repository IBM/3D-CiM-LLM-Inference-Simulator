import torch
from torch import fx

from ..utils import digital_add, get_logger
from .activation import get_activation_fn
from .base import BaseModule
from .embedding import Embedding
from .layernorm import LayerNorm
from .linear import Linear
from .moe import MoELayer
from .multihead_attention import MultiheadAttention

fx.wrap(digital_add)


class TransformerEncoder(BaseModule, torch.nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers,
        add_embedding_layer: bool = False,
        embedding_layer_kwargs: dict = {},
        encoder_layer_kwargs: dict = {},
        moe_kwargs: dict = {},
        device="meta",
        dtype=None,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.logger = get_logger("Encoder")
        self.layers = self._get_clones(
            encoder_layer,
            num_layers,
            encoder_layer_kwargs,
            add_embedding_layer,
            embedding_layer_kwargs,
            moe_kwargs,
            **self.factory_kwargs,
        )
        self.num_layers = num_layers

    def forward(
        self, src: torch.Tensor, seq_length: int, op_info: dict = {}
    ) -> torch.Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, seq_length, op_info)

        return output

    def _get_clones(
        self,
        encoder_layer: torch.nn.Module,
        num_layers: int,
        encoder_layer_kwargs: dict,
        add_embedding_layer: bool,
        embedding_layer_kwargs: dict,
        moe_kwargs: dict,
        device: str,
        dtype,
    ) -> torch.nn.ModuleList:
        frequency = 0
        # if we even want an MoE
        if moe_kwargs != {}:
            assert (
                "frequency" in moe_kwargs
            ), "key 'frequency' not found. Must be specified and > 0"
            frequency = moe_kwargs["frequency"]
            assert (
                frequency > 0
            ), "'frequency' must be > 0. A layer has experts if layer_idx % frequency == 0"
            # note: this means the first layer is always an moe layer

        return torch.nn.ModuleList(
            [
                encoder_layer(
                    is_first=i == 0,
                    is_last=i == num_layers - 1,
                    do_embedding=(i == 0 and add_embedding_layer),
                    embedding_layer_kwargs=embedding_layer_kwargs,
                    moe_kwargs=moe_kwargs
                    if frequency > 0 and i % frequency == 0
                    else {},
                    **encoder_layer_kwargs,
                    **{"device": device, "dtype": dtype},
                )
                for i in range(num_layers)
            ]
        )


class TransformerEncoderLayer(BaseModule, torch.nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        is_first: bool,
        is_last: bool,
        dim_feedforward: int = 2048,
        activation: str = "relu",
        norm_first: bool = False,
        layer_norm_eps: float = 1e-5,
        do_embedding: bool = False,
        embedding_layer_kwargs: dict = {},
        moe_kwargs: dict = {},
        device="meta",
        dtype=None,
    ):
        assert not norm_first, "norm_first must be False"
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.logger = get_logger("EncoderLayer")
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")

        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.dim_feedforward = dim_feedforward
        self.is_first = is_first
        self.is_last = is_last
        self.do_embedding = do_embedding
        assert (
            self.head_dim * nhead == self.d_model
        ), "d_model must be divisible by num_heads"

        if self.is_first:
            if do_embedding:
                assert (
                    "vocab_size" in embedding_layer_kwargs
                    and "max_seq_length" in embedding_layer_kwargs
                )
                self.token_embedding = Embedding(
                    embedding_layer_kwargs["vocab_size"],
                    embedding_layer_kwargs["embedding_dim"],
                    **self.factory_kwargs,
                )
                self.pos_embedding = Embedding(
                    embedding_layer_kwargs["max_seq_length"],
                    embedding_layer_kwargs["embedding_dim"],
                    **self.factory_kwargs,
                )
            self.q_proj_in = Linear(d_model, d_model, **self.factory_kwargs)
            self.k_proj_in = Linear(d_model, d_model, **self.factory_kwargs)
            self.v_proj_in = Linear(d_model, d_model, **self.factory_kwargs)

        if not self.is_last:
            self.q_proj_out = Linear(d_model, d_model, **self.factory_kwargs)
            self.k_proj_out = Linear(d_model, d_model, **self.factory_kwargs)
            self.v_proj_out = Linear(d_model, d_model, **self.factory_kwargs)

        self.out_proj = Linear(d_model, d_model, **self.factory_kwargs)

        self.is_sparse = False
        if moe_kwargs == {}:
            # Implementation of Feedforward model
            self.ffn1 = Linear(d_model, dim_feedforward, **self.factory_kwargs)
            self.ffn2 = Linear(dim_feedforward, d_model, **self.factory_kwargs)
        else:
            self.is_sparse = True
            moe_kwargs["d_model"] = d_model
            moe_kwargs["dim_feedforward"] = dim_feedforward
            self.moe_layer = MoELayer(**moe_kwargs, **self.factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **self.factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **self.factory_kwargs)

        self.self_attn = MultiheadAttention(self.d_model)

        assert isinstance(activation, str), "activation must be string"
        self.activation = get_activation_fn(activation)

        if self.is_last:
            self.last_norm = LayerNorm(
                d_model, eps=layer_norm_eps, **self.factory_kwargs
            )

    def forward(
        self,
        input: torch.Tensor | tuple[torch.Tensor],
        seq_length: int,
        op_info: dict = {},
    ):
        if self.is_first:
            q_buf, k_buf, v_buf, embed_buf = self._get_buffer(
                [seq_length, seq_length, seq_length, seq_length]
            )
            for token_idx in range(seq_length):
                op_info = {**op_info, "token_id": token_idx, "seq_len": seq_length}
                if self.do_embedding:
                    embedded_input = digital_add(
                        self.token_embedding(
                            input[token_idx],
                            op_info=op_info,
                        ),
                        self.pos_embedding(
                            token_idx,
                            op_info=op_info,
                        ),
                        op_info={**op_info, "size": self.d_model},
                    )
                else:
                    embedded_input = input[token_idx]
                # OOO execution not supported (for one buffer). QKV can be OOO.
                q_buf = torch.cat(
                    [
                        q_buf,
                        self.q_proj_in(
                            embedded_input,
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )
                k_buf = torch.cat(
                    [
                        k_buf,
                        self.k_proj_in(
                            embedded_input,
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )
                v_buf = torch.cat(
                    [
                        v_buf,
                        self.v_proj_in(
                            embedded_input,
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )
                embed_buf = torch.cat([embed_buf, embedded_input.view(1, -1)])
            prev_layer_input = embed_buf
        else:
            q_buf, k_buf, v_buf, prev_layer_input = input

        # sequence
        q = q_buf.view(seq_length, self.nhead, self.head_dim).transpose(0, 1)
        k = k_buf.view(seq_length, self.nhead, self.head_dim).transpose(0, 1)
        v = v_buf.view(seq_length, self.nhead, self.head_dim).transpose(0, 1)

        q = q.view(1, self.nhead, seq_length, self.head_dim)
        k = k.view(1, self.nhead, seq_length, self.head_dim)
        v = v.view(1, self.nhead, seq_length, self.head_dim)

        attn_output = self.self_attn(
            q,
            k,
            v,
            seq_length,
            None,  # if we don't pass an attn_mask, we are in encoder mode
            op_info={
                "seq_len": seq_length,
                "decoding_id": op_info.get("decoding_id", None),
                "size": self.d_model,
                "causal": False,
                "nhead": self.nhead,
            },
        )

        if self.is_last:
            output_buffer = self._get_buffer(seq_length)
        else:
            q_buf, k_buf, v_buf, next_layer_residual_input_buf = self._get_buffer(
                [seq_length, seq_length, seq_length, seq_length]
            )

        for token_idx in range(seq_length):
            op_info = {
                "token_id": token_idx,
                "seq_len": seq_length,
                "decoding_id": op_info.get("decoding_id", None),
            }
            x = self.out_proj(
                attn_output[token_idx],
                op_info=op_info,
            )
            _x = digital_add(
                prev_layer_input[token_idx],
                x,
                op_info={"size": self.d_model, **op_info},
            )
            x = self.norm1(_x, op_info={"size": self.d_model, **op_info})

            _x = digital_add(
                x,
                self._ff_block(x, op_info=op_info),
                op_info={"size": self.d_model, **op_info},
            )
            x = self.norm2(_x, op_info={"size": self.d_model, **op_info})

            if self.is_last:
                output_buffer = torch.cat([output_buffer, x.view(1, -1)])
            else:
                q_buf = torch.cat(
                    [
                        q_buf,
                        self.q_proj_out(x, op_info=op_info).view(1, -1),
                    ]
                )
                k_buf = torch.cat(
                    [
                        k_buf,
                        self.k_proj_out(x, op_info=op_info).view(1, -1),
                    ]
                )
                v_buf = torch.cat(
                    [
                        v_buf,
                        self.v_proj_out(x, op_info=op_info).view(1, -1),
                    ]
                )
                # Wrong, that way we create dependencies across tokens on the residual of the next layer
                next_layer_residual_input_buf = torch.cat(
                    [next_layer_residual_input_buf, x.view(1, -1)]
                )

        if self.is_last:
            return output_buffer
        else:
            return q_buf, k_buf, v_buf, next_layer_residual_input_buf

    def _get_buffer(self, lens: list[int] | int):
        if isinstance(lens, int):
            lens = [lens]
        bufs = [torch.tensor([], **self.factory_kwargs) for _ in lens]
        if len(bufs) == 1:
            bufs = bufs[0]
        return bufs

    def _ff_block(self, input: torch.Tensor, op_info: dict = None):
        if self.is_sparse:
            return self.moe_layer(input, op_info=op_info)

        x = self.ffn1(input, op_info=op_info)
        x = self.activation(x, op_info={"size": self.dim_feedforward, **op_info})
        x = self.ffn2(x, op_info=op_info)
        return x
