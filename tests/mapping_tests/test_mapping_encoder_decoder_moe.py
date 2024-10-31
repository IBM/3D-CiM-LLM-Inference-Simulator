import torch

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.mapping import Mapper, MapStrategy, Strategy
from threedsim.models import EncoderDecoderTransformer
from threedsim.modules import TransformerDecoderLayer, TransformerEncoderLayer
from threedsim.modules.base import assign_acc


def test_mapping():
    for strategy in [
        Strategy.GREEDY_IN_ORDER,
        Strategy.GREEDY_RANDOM,
        Strategy.GREEDY_TOP2,
        Strategy.GREEDY_UTIL,
        Strategy.MOE_SEPARATED,
    ]:
        config = AcceleratorConfig(tiles=16, tiers=21, tier_shape=(512, 512))

        context_len = 4
        gen_target_len = 4

        device = "meta"
        acc = Accelerator(config, device=device)
        enc_dec_layer_kwargs = {
            "d_model": 768,
            "nhead": 8,
            "dim_feedforward": 4 * 768,
        }
        embedding_layer_kwargs = {
            "vocab_size": 2048,
            "embedding_dim": 768,
            "max_seq_length": max(gen_target_len, context_len),
        }
        moe_kwargs = {"frequency": 2, "num_experts": 3, "k": 2}
        model = EncoderDecoderTransformer(
            TransformerEncoderLayer,
            TransformerDecoderLayer,
            num_encoder_layers=2,
            num_decoder_layers=2,
            encoder_layer_kwargs=enc_dec_layer_kwargs,
            decoder_layer_kwargs=enc_dec_layer_kwargs,
            embedding_layer_kwargs=embedding_layer_kwargs,
            moe_kwargs=moe_kwargs,
            share_embedding=False,
            device=device,
        )

        assign_acc(model, acc)

        mapper = Mapper(
            accelerator=acc,
            model=model,
            map_strategy=MapStrategy(
                strategy=strategy, split_ffn=True, stack_embedding=True
            ),
            density=torch.rand(3, 3),
        )
        mapper.map_network()

        bsz = 1
        src_len = 10
        idx = torch.randint(low=0, high=2048, size=(bsz, src_len))
        model(
            context_src=idx[0],
            gen_src=torch.tensor(0)
            .view(
                -1,
            )
            .long(),
            context_len=src_len,
            gen_start_len=1,
            gen_target_len=1 + 3,
        )


if __name__ == "__main__":
    test_mapping()
