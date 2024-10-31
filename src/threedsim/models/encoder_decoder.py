from typing import Optional

import torch
from torch import fx

from ..modules.base import BaseModule
from ..modules.decoder import TransformerDecoder
from ..modules.encoder import TransformerEncoder
from ..modules.layernorm import LayerNorm
from ..modules.linear import Linear
from ..utils import cross_dependency, get_logger, multinomial

fx.wrap(cross_dependency)
fx.wrap(multinomial)


class EncoderDecoderTransformer(BaseModule, torch.nn.Module):
    def __init__(
        self,
        encoder_layer,
        decoder_layer,
        num_encoder_layers: int,
        num_decoder_layers: int,
        embedding_layer_kwargs: dict = {},
        moe_kwargs: dict = {},
        encoder_layer_kwargs: dict = {},
        decoder_layer_kwargs: dict = {},
        share_embedding: bool = False,
        token_density: Optional[torch.Tensor] = None,
        device="meta",
        dtype=None,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.logger = get_logger("EncoderDecoderTransformer")

        # each token has equal probability
        if token_density is None and device == "meta":
            token_density = torch.rand(embedding_layer_kwargs["vocab_size"])
            token_density /= token_density.sum()

        self.token_density = token_density
        self.d_model = decoder_layer_kwargs["d_model"]
        self.vocab_size = embedding_layer_kwargs["vocab_size"]
        self.encoder_stack = TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers,
            add_embedding_layer=True,
            embedding_layer_kwargs=embedding_layer_kwargs,
            moe_kwargs=moe_kwargs,
            encoder_layer_kwargs=encoder_layer_kwargs,
            **self.factory_kwargs,
        )
        decoder_layer_kwargs["with_memory"] = True
        self.decoder_stack = TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers,
            add_embedding_layer=True,
            embedding_layer_kwargs=embedding_layer_kwargs,
            moe_kwargs=moe_kwargs,
            decoder_layer_kwargs=decoder_layer_kwargs,
            **self.factory_kwargs,
        )
        self.norm = LayerNorm(self.d_model, **self.factory_kwargs)
        self.lm_head = Linear(self.d_model, self.vocab_size, **self.factory_kwargs)

        if share_embedding:
            self._share_embedding_layer()

    def forward(
        self,
        context_src: torch.Tensor,
        gen_src: torch.Tensor,
        context_len: int,
        gen_start_len: int,
        gen_target_len: int,
    ):
        memory = self.encoder_stack(
            context_src, seq_length=context_len, op_info={"decoding_id": "memory"}
        )
        # Add dependency between memory and X
        x, memory = cross_dependency(gen_src, memory)
        for decoding_id, seq_length in enumerate(range(gen_start_len, gen_target_len)):
            x = self.decoder_stack(
                gen_src,
                seq_length,
                memory=memory,
                memory_len=context_len,
                op_info={"decoding_id": decoding_id},
            )
            token_of_interest = _index_sequence(x, -1)
            token_of_interest = self.norm(
                token_of_interest,
                op_info={
                    "size": self.d_model,
                    "token_id": seq_length - 1,
                    "seq_len": seq_length,
                    "decoding_id": decoding_id,
                },
            )
            logits = self.lm_head(
                token_of_interest,
                op_info={
                    "token_id": seq_length - 1,
                    "seq_len": seq_length,
                    "decoding_id": decoding_id,
                },
            )

            if self.factory_kwargs["device"] == "meta":
                assert (
                    self.token_density is not None
                ), "device is meta but token_density is not specified"
                # multinomial is not implemented for meta device
                probs, logits = cross_dependency(self.token_density, logits)
            else:
                probs = torch.nn.functional.softmax(logits, dim=-1)

            next_token = multinomial(
                probs,
                num_samples=1,
                op_info={
                    "size": 512,  # Fake to show that it is outputing a one-hot for the embeddings
                    "vocab_size": self.vocab_size,
                    "token_id": seq_length - 1,
                    "seq_len": seq_length,
                    "decoding_id": decoding_id,
                },
            )
            gen_src = torch.cat((gen_src, next_token))

        return gen_src

    # Share the mapping info between the two embedding layers
    def _share_embedding_layer(self):
        self.decoder_stack.layers[0].token_embedding.mapping = (
            self.encoder_stack.layers[0].token_embedding.mapping
        )
        self.decoder_stack.layers[0].token_embedding.n_vertical = (
            self.encoder_stack.layers[0].token_embedding.n_vertical
        )
        self.decoder_stack.layers[0].token_embedding.n_horizontal = (
            self.encoder_stack.layers[0].token_embedding.n_horizontal
        )

        self.decoder_stack.layers[0].pos_embedding.mapping = self.encoder_stack.layers[
            0
        ].pos_embedding.mapping
        self.decoder_stack.layers[0].pos_embedding.n_vertical = (
            self.encoder_stack.layers[0].pos_embedding.n_vertical
        )
        self.decoder_stack.layers[0].pos_embedding.n_horizontal = (
            self.encoder_stack.layers[0].pos_embedding.n_horizontal
        )


@torch.fx.wrap
def _index_sequence(x: torch.Tensor, idx: int):
    return x[idx]
