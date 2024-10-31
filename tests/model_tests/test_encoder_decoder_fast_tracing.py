from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.inference import (
    schedule_execution,
    trace_model,
    fast_trace_encoder_decoder,
)
from threedsim.models import EncoderDecoderTransformer
from threedsim.modules import TransformerEncoderLayer, TransformerDecoderLayer
from threedsim.modules.base import (
    assign_acc,
    fill_name_fields,
    make_traceable,
    make_use_linear,
)
from threedsim.mapping import Mapper, MapStrategy, Strategy


def test_encoder_decoder_fast_tracing():
    config = AcceleratorConfig(tiles=10, tiers=21, tier_shape=(512, 512))

    num_sequences = 1
    context_len = 3
    gen_start_len = 1
    gen_target_len = 4

    device = "meta"
    acc = Accelerator(config, device=device)
    enc_dec_layer_kwargs = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 512,
    }
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": max(gen_target_len, context_len),
    }
    model = EncoderDecoderTransformer(
        TransformerEncoderLayer,
        TransformerDecoderLayer,
        num_encoder_layers=2,
        num_decoder_layers=2,
        encoder_layer_kwargs=enc_dec_layer_kwargs,
        decoder_layer_kwargs=enc_dec_layer_kwargs,
        embedding_layer_kwargs=embedding_layer_kwargs,
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
    )
    mapper.map_network()
    fill_name_fields(model)
    make_traceable(model, is_traceable=True)

    symbolic_traced = trace_model(
        model,
        num_sequences,
        concrete_args={
            "context_len": context_len,
            "gen_start_len": gen_start_len,
            "gen_target_len": gen_target_len,
        },
    )

    (
        standard_execution_time,
        standard_memory,
        standard_peak_memory,
        standard_energy,
        _,
        _,
        _,
    ) = schedule_execution(
        symbolic_traced.graph,
        accelerator=model.accelerator,
        communication=True,
    )
    print(f"Execution took {standard_execution_time} units")
    print(f"Required {standard_peak_memory} bytes of scratchpad memory")
    print(f"Spent {standard_energy} xJ of energy")

    # New fast tracing
    make_use_linear(model, use_linear=True)
    fast_traced = fast_trace_encoder_decoder(
        model,
        context_len=context_len,
        gen_start_len=gen_start_len,
        gen_target_len=gen_target_len,
        bsz=num_sequences,
    )
    execution_time, memory, peak_memory, energy, flops, _, _ = schedule_execution(
        fast_traced.graph,
        accelerator=model.accelerator,
        copy_and_cleanup_graph=False,
        communication=True,
    )
    print(f"Execution took {execution_time} units")
    print(f"Required {peak_memory} bytes of scratchpad memory")
    print(f"Spent {energy} xJ of energy")

    assert execution_time == standard_execution_time, "Execution time is not correct"
    assert energy == standard_energy, "Energy is not correct"
    assert max(memory) == max(standard_memory), "Memory is not correct"


if __name__ == "__main__":
    test_encoder_decoder_fast_tracing()
