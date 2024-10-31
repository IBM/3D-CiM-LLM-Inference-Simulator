import torch

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.mapping import Mapper, MapStrategy, Strategy
from threedsim.models import EncoderDecoderTransformer
from threedsim.modules import TransformerDecoderLayer, TransformerEncoderLayer
from threedsim.modules.base import assign_acc
from threedsim.inference import schedule_execution, trace_model
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable


def test_model_sram():
    for model_sram in [True, False]:
        config = AcceleratorConfig(
            tiles=16, tiers=21, tier_shape=(512, 512), model_sram=model_sram
        )

        context_len = 4
        gen_start_len = 2
        gen_target_len = 4
        num_sequences = 1

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
                strategy=Strategy.GREEDY_IN_ORDER, split_ffn=True, stack_embedding=True
            ),
            density=torch.rand(3, 3),
        )
        mapper.map_network()
        concrete_args = {
            "context_len": context_len,
            "gen_start_len": gen_start_len,
            "gen_target_len": gen_target_len,
        }
        fill_name_fields(model)
        make_traceable(model, is_traceable=True)
        symbolic_traced = trace_model(model, num_sequences, concrete_args=concrete_args)
        execution_time, memory, peak_memory, energy, flops, _, _ = schedule_execution(
            symbolic_traced.graph,
            accelerator=model.accelerator,
            plot=True,
            plot_dir="results",
        )
        print(f"Execution took {execution_time} units")
        print(f"Required {peak_memory} bytes of scratchpad memory")
        print(f"Spent {energy} xJ of energy")


if __name__ == "__main__":
    test_model_sram()
