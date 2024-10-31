from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.modules import TransformerEncoderLayer
from threedsim.utils import set_seed

from threedsim.inference import schedule_execution, trace_model
from threedsim.models import EncoderOnlyTransformer
from threedsim.modules.base import (
    assign_acc,
    fill_name_fields,
    make_traceable,
)
from threedsim.mapping import Mapper, MapStrategy, Strategy


def test_bert():
    set_seed(0)

    d_model = 768
    num_sequences = 1
    context_len = 32
    num_layers = 12

    config = AcceleratorConfig(
        mvm_latency=40,
        num_digital_units=288,
        num_mha_units=96,
        tiles=500,
        tiers=1,
        tier_shape=(512, 512),
    )
    acc = Accelerator(config, device="meta")
    encoder_layer_kwargs = {
        "d_model": d_model,
        "nhead": 12,
        "dim_feedforward": 4 * d_model,
    }
    embedding_layer_kwargs = {
        "vocab_size": 512,
        "embedding_dim": d_model,
        "max_seq_length": context_len,
    }
    model = EncoderOnlyTransformer(
        TransformerEncoderLayer,
        num_layers=num_layers,
        encoder_layer_kwargs=encoder_layer_kwargs,
        embedding_layer_kwargs=embedding_layer_kwargs,
        device="meta",
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
        concrete_args={"context_len": context_len},
    )
    (
        execution_time,
        memory,
        peak_memory,
        energy,
        flops,
        _,
        _,
    ) = schedule_execution(
        symbolic_traced.graph,
        accelerator=model.accelerator,
        plot=False,
        plot_dir="results",
        communication=True,
    )

    print(f"Execution took {execution_time:,} ns")
    print(f"Required {peak_memory:,} bytes of scratchpad memory")
    print(f"Spent {energy:,} nJ of energy")
    print(f"Total number of TFLOPs {flops/1e12:,}")
    print(f"Total number of FLOPs {flops:,}")


if __name__ == "__main__":
    test_bert()
