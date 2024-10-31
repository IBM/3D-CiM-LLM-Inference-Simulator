import torch

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.mapping import Mapper, MapStrategy, Strategy
from threedsim.models import DecoderOnlyTransformer
from threedsim.modules import TransformerDecoderLayer
from threedsim.modules.base import assign_acc


def test_mapping():
    for strategy in [
        Strategy.GREEDY_IN_ORDER,
        Strategy.GREEDY_RANDOM,
        Strategy.GREEDY_UTIL,
        Strategy.MOE_SEPARATED,
    ]:
        config = AcceleratorConfig(tiles=96, tiers=257, tier_shape=(512, 512))

        context_len = 1
        gen_target_len = 8

        device = "meta"
        acc = Accelerator(config, device=device)
        layer_kwargs = {
            "d_model": 1024,
            "nhead": 8,
            "dim_feedforward": 4 * 1024,
        }
        embedding_layer_kwargs = {
            "vocab_size": 50000,
            "embedding_dim": 1024,
            "max_seq_length": max(gen_target_len, context_len),
        }
        moe_kwargs = {"frequency": 1, "num_experts": 64, "k": 1}
        model = DecoderOnlyTransformer(
            TransformerDecoderLayer,
            num_layers=10,
            decoder_layer_kwargs=layer_kwargs,
            embedding_layer_kwargs=embedding_layer_kwargs,
            moe_kwargs=moe_kwargs,
            device=device,
        )

        assign_acc(model, acc)

        mapper = Mapper(
            accelerator=acc,
            model=model,
            map_strategy=MapStrategy(
                strategy=strategy, split_ffn=True, stack_embedding=True
            ),
        )
        mapper.map_network()


if __name__ == "__main__":
    test_mapping()
