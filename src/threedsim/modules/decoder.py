import torch
from torch import fx

from ..utils import add_dependency, digital_add, get_logger
from .activation import get_activation_fn
from .base import BaseModule
from .embedding import Embedding
from .layernorm import LayerNorm
from .linear import Linear
from .moe import MoELayer
from .multihead_attention import MultiheadAttention

fx.wrap(digital_add)
fx.wrap(add_dependency)


class TransformerDecoder(BaseModule, torch.nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        add_embedding_layer: bool = False,
        embedding_layer_kwargs: dict = {},
        decoder_layer_kwargs: dict = {},
        moe_kwargs: dict = {},
        device="meta",
        dtype=None,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.logger = get_logger("Decoder")
        self.layers = self._get_clones(
            decoder_layer,
            num_layers,
            decoder_layer_kwargs,
            add_embedding_layer,
            embedding_layer_kwargs,
            moe_kwargs,
            self.factory_kwargs,
        )
        self.num_layers = num_layers

    def forward(
        self,
        src: torch.Tensor,
        seq_length: int,
        memory: tuple[torch.Tensor] | None = None,
        memory_len: int | None = None,
        op_info: dict = {},
    ) -> torch.Tensor:
        output = src

        for mod in self.layers:
            output = mod(output, seq_length, memory, memory_len, op_info)

        return output

    def _get_clones(
        self,
        decoder_layer: torch.nn.Module,
        num_layers: int,
        decoder_layer_kwargs: dict,
        add_embedding_layer: bool,
        embedding_layer_kwargs: dict,
        moe_kwargs: dict,
        factory_kwargs: dict,
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
                decoder_layer(
                    is_first=i == 0,
                    is_last=i == num_layers - 1,
                    do_embedding=(i == 0 and add_embedding_layer),
                    embedding_layer_kwargs=embedding_layer_kwargs,
                    moe_kwargs=moe_kwargs
                    if frequency > 0 and i % frequency == 0
                    else {},
                    **decoder_layer_kwargs,
                    **factory_kwargs,
                )
                for i in range(num_layers)
            ]
        )


class TransformerDecoderLayer(BaseModule, torch.nn.Module):
    __constants__ = ["norm_first"]

    def __init__(
        self,
        d_model: int,
        nhead: int,
        is_first: bool,
        is_last: bool,
        dim_feedforward: int = 2048,
        activation: str = "relu",
        layer_norm_eps: float = 1e-5,
        norm_first: bool = False,
        do_embedding: bool = False,
        embedding_layer_kwargs: dict = {},
        moe_kwargs: dict = {},
        with_memory: bool = False,
        device="meta",
        dtype=None,
    ) -> None:
        self.factory_kwargs = {"device": device, "dtype": dtype}
        self.logger = get_logger("DecoderLayer")
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
            moe_kwargs["d_model"] = d_model
            moe_kwargs["dim_feedforward"] = dim_feedforward
            self.is_sparse = True
            self.moe_layer = MoELayer(**moe_kwargs, **self.factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **self.factory_kwargs)
        self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **self.factory_kwargs)
        # norm3 is the norm for the next layer. if we are the last one,
        # we use it as the final layer norm in gpt2
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **self.factory_kwargs)

        self.self_attn = MultiheadAttention(self.d_model)
        msl = (
            512
            if not "max_seq_length" in embedding_layer_kwargs
            else embedding_layer_kwargs["max_seq_length"]
        )
        self.attn_mask = torch.tril(torch.ones(msl, msl, **self.factory_kwargs)).view(
            1, 1, msl, msl
        )
        if with_memory:
            self.cross_attn = MultiheadAttention(self.d_model)
            self.q_proj_cross = Linear(d_model, d_model, **self.factory_kwargs)
            self.k_proj_cross = Linear(d_model, d_model, **self.factory_kwargs)
            self.v_proj_cross = Linear(d_model, d_model, **self.factory_kwargs)
            self.out_proj_cross = Linear(d_model, d_model, **self.factory_kwargs)

        assert isinstance(activation, str), "activation must be string"
        self.activation = get_activation_fn(activation)

    def forward(
        self,
        input: torch.Tensor | tuple[torch.Tensor],
        seq_length: int,
        memory: tuple[torch.Tensor] | None = None,
        memory_len: int | None = None,
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

                if self.norm_first:
                    # gpt2: pass the input through layer norm
                    normed_embedded_input = self.norm1(
                        embedded_input, op_info={"size": self.d_model, **op_info}
                    )
                else:
                    normed_embedded_input = embedded_input

                # OOO execution not supported (for one buffer). QKV can be OOO.
                q_buf = torch.cat(
                    [
                        q_buf,
                        self.q_proj_in(
                            normed_embedded_input,
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )
                k_buf = torch.cat(
                    [
                        k_buf,
                        self.k_proj_in(
                            normed_embedded_input,
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )
                v_buf = torch.cat(
                    [
                        v_buf,
                        self.v_proj_in(
                            normed_embedded_input,
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )
                embed_buf = torch.cat([embed_buf, embedded_input.view(1, -1)])

            prev_layer_input = embed_buf
        else:
            q_buf, k_buf, v_buf, prev_layer_input = input

        # sequence
        q = (
            q_buf.view(seq_length, self.nhead, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        k = (
            k_buf.view(seq_length, self.nhead, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )
        v = (
            v_buf.view(seq_length, self.nhead, self.head_dim)
            .transpose(0, 1)
            .contiguous()
        )

        q = q.view(1, self.nhead, seq_length, self.head_dim)
        k = k.view(1, self.nhead, seq_length, self.head_dim)
        v = v.view(1, self.nhead, seq_length, self.head_dim)

        # Self attention
        attn_output = self.self_attn(
            q,
            k,
            v,
            seq_length,
            self.attn_mask,
            op_info={
                "seq_len": seq_length,
                "decoding_id": op_info.get("decoding_id", None),
                "size": self.d_model,
                "causal": True,
                "nhead": self.nhead,
            },
        )

        if memory is not None:
            assert memory_len is not None
            # Add a dependency between memory and the query to track them
            q_cross_buf, k_cross_buf, v_cross_buf, resid_buffer = self._get_buffer(
                [memory_len, memory_len, memory_len, memory_len]
            )
            for token_idx in range(memory_len):
                op_info = {
                    "token_id": token_idx,
                    "seq_len": memory_len,
                    "decoding_id": op_info.get("decoding_id", None),
                }
                # OOO execution not supported (for one buffer). QKV can be OOO.
                if token_idx < seq_length:
                    x = self.out_proj(
                        attn_output[token_idx],
                        op_info={
                            "token_id": token_idx,
                            "seq_len": seq_length,
                            "decoding_id": op_info.get("decoding_id", None),
                        },
                    )
                    x, memory = add_dependency(x, memory)
                    x = digital_add(
                        x,
                        prev_layer_input[token_idx],
                        op_info={
                            "size": self.d_model,
                            "token_id": token_idx,
                            "seq_len": seq_length,
                            "decoding_id": op_info.get("decoding_id", None),
                        },
                    )
                    if not self.norm_first:
                        x = self.norm1(
                            x,
                            op_info={
                                "size": self.d_model,
                                "token_id": token_idx,
                                "seq_len": seq_length,
                                "decoding_id": op_info.get("decoding_id", None),
                            },
                        )
                        resid_buffer = torch.cat([resid_buffer, x.view(1, -1)])

                    q_cross_buf = torch.cat(
                        [
                            q_cross_buf,
                            self.q_proj_cross(
                                x,
                                op_info=op_info,
                            ).view(1, -1),
                        ]
                    )

                k_cross_buf = torch.cat(
                    [
                        k_cross_buf,
                        self.k_proj_cross(
                            memory[token_idx],
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )
                v_cross_buf = torch.cat(
                    [
                        v_cross_buf,
                        self.v_proj_cross(
                            memory[token_idx],
                            op_info=op_info,
                        ).view(1, -1),
                    ]
                )

            q = (
                q_cross_buf.view(seq_length, self.nhead, self.head_dim)
                .transpose(0, 1)
                .contiguous()
            )
            k = (
                k_cross_buf.view(memory_len, self.nhead, self.head_dim)
                .transpose(0, 1)
                .contiguous()
            )
            v = (
                v_cross_buf.view(memory_len, self.nhead, self.head_dim)
                .transpose(0, 1)
                .contiguous()
            )

            q = q.view(1, self.nhead, seq_length, self.head_dim)
            k = k.view(1, self.nhead, memory_len, self.head_dim)
            v = v.view(1, self.nhead, memory_len, self.head_dim)

            cross_attn_output = self.cross_attn(
                q,
                k,
                v,
                seq_length,
                None,
                op_info={
                    "seq_len": seq_length,
                    "decoding_id": op_info.get("decoding_id", None),
                    "size": self.d_model,
                    "causal": False,
                    "nhead": self.nhead,
                },
            )
            cross_attn_output = _index_subsequence(cross_attn_output, slice(seq_length))

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
            if memory is None:
                x = self.out_proj(
                    attn_output[token_idx],
                    op_info=op_info,
                )
                # gpt2 style: prev_layer_input is the input before the first layer norm
                x = digital_add(
                    x,
                    prev_layer_input[token_idx],
                    op_info={"size": self.d_model, **op_info},
                )
            else:
                x = self.out_proj_cross(
                    cross_attn_output[token_idx],
                    op_info=op_info,
                )
                # residual add
                _x = digital_add(
                    x,
                    resid_buffer[token_idx],
                    op_info={"size": self.d_model, **op_info},
                )
                x = self.norm2(_x, op_info={"size": self.d_model, **op_info})

            if memory is not None:
                ff_out = self._ff_block(x, op_info=op_info)
            else:
                # gpt2, pass x through norm2 and then through the ff block
                ff_out = self._ff_block(
                    self.norm2(x, op_info={"size": self.d_model, **op_info}),
                    op_info=op_info,
                )

            x = digital_add(
                ff_out, x, op_info={"size": self.d_model, **op_info}
            )  # residual

            if self.is_last:
                if memory is not None:
                    x = self.norm3(
                        x, op_info={"size": self.d_model, **op_info}
                    )  # torch decoder by default applies norm after last ffn
                output_buffer = torch.cat([output_buffer, x.view(1, -1)])
            else:
                if memory is not None:
                    next_layer_residual_input = x
                else:
                    next_layer_residual_input = x.clone()
                    # norm3 is used for the layer norm of the next layer, but in gpt, the residual branches off
                    # before https://commons.wikimedia.org/wiki/File:Full_GPT_architecture.png
                    x = self.norm3(x, op_info={"size": self.d_model, **op_info})

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
                next_layer_residual_input_buf = torch.cat(
                    [next_layer_residual_input_buf, next_layer_residual_input]
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


@torch.fx.wrap
def _index_subsequence(sequence: torch.Tensor, slice: slice):
    return sequence[slice]
