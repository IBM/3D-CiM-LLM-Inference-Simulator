from typing import Optional

import torch
from torch import fx

from ..modules.base import BaseModule
from ..modules.decoder import TransformerDecoder
from ..modules.layernorm import LayerNorm
from ..modules.linear import Linear
from ..utils import get_logger, cross_dependency, multinomial

fx.wrap(cross_dependency)
fx.wrap(multinomial)


class DecoderOnlyTransformer(BaseModule, torch.nn.Module):
    def __init__(
        self,
        decoder_layer,
        num_layers,
        embedding_layer_kwargs: dict = {},
        decoder_layer_kwargs: dict = {},
        moe_kwargs: dict = {},
        token_density: Optional[torch.Tensor] = None,
        device="meta",
        dtype=None,
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.logger = get_logger("DecoderOnlyTransformer")

        # each token has equal probability
        if token_density is None and device == "meta":
            token_density = torch.rand(embedding_layer_kwargs["vocab_size"])
            token_density /= token_density.sum()

        self.token_density = token_density

        self.d_model = decoder_layer_kwargs["d_model"]
        self.vocab_size = embedding_layer_kwargs["vocab_size"]
        self.decoder_stack = TransformerDecoder(
            decoder_layer,
            num_layers=num_layers,
            add_embedding_layer=True,
            embedding_layer_kwargs=embedding_layer_kwargs,
            decoder_layer_kwargs=decoder_layer_kwargs,
            moe_kwargs=moe_kwargs,
            **self.factory_kwargs,
        )
        self.norm = LayerNorm(self.d_model, **self.factory_kwargs)
        self.lm_head = Linear(self.d_model, self.vocab_size, **self.factory_kwargs)

    def forward(
        self,
        src: torch.Tensor,
        start_len: int,
        target_len: int,
    ):
        for decoding_id, seq_length in enumerate(range(start_len, target_len)):
            x = self.decoder_stack(
                src, seq_length, op_info={"decoding_id": decoding_id}
            )
            token_of_interest = _index_subsequence(x, -1)
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

            src = torch.cat((src, next_token))

        return src


@torch.fx.wrap
def _index_subsequence(sequence: torch.Tensor, slice: slice):
    return sequence[slice]
