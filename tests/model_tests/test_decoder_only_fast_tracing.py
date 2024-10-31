from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.inference import schedule_execution, trace_model, fast_trace_decoder
from threedsim.models import DecoderOnlyTransformer
from threedsim.modules import TransformerDecoderLayer
from threedsim.modules.base import (
    assign_acc,
    fill_name_fields,
    make_traceable,
    make_use_linear,
)
from threedsim.mapping import Mapper, MapStrategy, Strategy


def test_decoder_only():
    config = AcceleratorConfig(tiles=100, tiers=1024, tier_shape=(512, 512))

    num_sequences = 3
    start_len = 1
    target_len = 3
    num_layers = 3
    d_model = 512
    d_ff = 4 * d_model
    vocab_size = 1024

    device = "meta"
    acc = Accelerator(config, device=device)
    decoder_layer_kwargs = {
        "d_model": d_model,
        "nhead": 8,
        "dim_feedforward": d_ff,
    }
    embedding_layer_kwargs = {
        "vocab_size": vocab_size,
        "embedding_dim": d_model,
        "max_seq_length": target_len,
    }
    model = DecoderOnlyTransformer(
        TransformerDecoderLayer,
        num_layers=num_layers,
        decoder_layer_kwargs=decoder_layer_kwargs,
        embedding_layer_kwargs=embedding_layer_kwargs,
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

    # Standard tracing
    symbolic_traced = trace_model(
        model,
        num_sequences,
        concrete_args={"start_len": start_len, "target_len": target_len},
    )
    (
        standard_execution_time,
        standard_memory,
        standard_peak_memory,
        standard_energy,
        flops,
        _,
        _,
    ) = schedule_execution(
        symbolic_traced.graph,
        accelerator=model.accelerator,
        plot=True,
        plot_dir="results",
        communication=True,
    )
    print(f"Execution took {standard_execution_time} units")
    print(f"Required {standard_peak_memory} bytes of scratchpad memory")
    print(f"Spent {standard_energy} xJ of energy")

    # New fast tracing
    make_use_linear(model, use_linear=True)
    fast_traced = fast_trace_decoder(
        model, start_len=start_len, target_len=target_len, bsz=num_sequences
    )
    execution_time, memory, peak_memory, energy, flops, _, _ = schedule_execution(
        fast_traced.graph,
        accelerator=model.accelerator,
        copy_and_cleanup_graph=False,
        plot=True,
        plot_dir="results",
        communication=True,
    )
    print(f"Execution took {execution_time} units")
    print(f"Required {peak_memory} bytes of scratchpad memory")
    print(f"Spent {energy} xJ of energy")

    assert execution_time == standard_execution_time, "Execution time is not correct"
    assert energy == standard_energy, "Energy is not correct"


if __name__ == "__main__":
    test_decoder_only()
