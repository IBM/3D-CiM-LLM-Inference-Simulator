from typing import Optional

import torch
from torch import fx

from ..modules.base import BaseModule
from ..modules.encoder import TransformerEncoder
from ..utils import cross_dependency, get_logger

fx.wrap(cross_dependency)


class EncoderOnlyTransformer(BaseModule, torch.nn.Module):
    def __init__(
        self,
        encoder_layer,
        num_layers: int,
        embedding_layer_kwargs: dict = {},
        moe_kwargs: dict = {},
        encoder_layer_kwargs: dict = {},
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
        self.d_model = encoder_layer_kwargs["d_model"]
        self.vocab_size = embedding_layer_kwargs["vocab_size"]
        self.encoder_stack = TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            add_embedding_layer=False,
            embedding_layer_kwargs=embedding_layer_kwargs,
            moe_kwargs=moe_kwargs,
            encoder_layer_kwargs=encoder_layer_kwargs,
            **self.factory_kwargs,
        )

    def forward(self, context_src: torch.Tensor, context_len: int):
        memory = self.encoder_stack(
            context_src, seq_length=context_len, op_info={"decoding_id": "memory"}
        )
        return memory
