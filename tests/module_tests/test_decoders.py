import torch

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.graph import prune_and_merge_ops, remove_unimportant_nodes_and_reconnect
from threedsim.inference import schedule_execution, trace_model
from threedsim.modules import TransformerDecoder, TransformerDecoderLayer
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable
from threedsim.plotting import plot_graph

device = "meta"


def get_acc_and_kwargs():
    config = AcceleratorConfig(tiles=10, tiers=21, tier_shape=(512, 512))
    acc = Accelerator(config, device=device)
    decoder_layer_kwargs = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 512,
    }
    return acc, decoder_layer_kwargs


def test_decoder_stack():
    acc, decoder_layer_kwargs = get_acc_and_kwargs()
    seq_length = 5

    model = TransformerDecoder(
        TransformerDecoderLayer,
        num_layers=2,
        decoder_layer_kwargs=decoder_layer_kwargs,
        device=device,
    )

    assign_acc(model, acc)

    model.layers[0].k_proj_in.set_mapping([[(0, 1, 1, 512, 512)]])
    model.layers[0].q_proj_in.set_mapping([[(0, 2, 1, 512, 512)]])
    model.layers[0].v_proj_in.set_mapping([[(0, 3, 1, 512, 512)]])
    model.layers[0].out_proj.set_mapping([[(0, 4, 1, 512, 512)]])
    model.layers[0].ffn1.set_mapping([[(0, 5, 1, 512, 512)]])
    model.layers[0].ffn2.set_mapping([[(0, 6, 1, 512, 512)]])

    model.layers[0].k_proj_out.set_mapping([[(1, 1, 1, 512, 512)]])
    model.layers[0].q_proj_out.set_mapping([[(1, 2, 1, 512, 512)]])
    model.layers[0].v_proj_out.set_mapping([[(1, 3, 1, 512, 512)]])
    model.layers[1].out_proj.set_mapping([[(1, 4, 1, 512, 512)]])
    model.layers[1].ffn1.set_mapping([[(1, 5, 1, 512, 512)]])
    model.layers[1].ffn2.set_mapping([[(1, 6, 1, 512, 512)]])
    concrete_args = {"seq_length": seq_length, "memory": None}
    return model, concrete_args


def test_decoder_stack_with_embeddings():
    acc, decoder_layer_kwargs = get_acc_and_kwargs()
    seq_length = 5
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": seq_length,
    }
    model = TransformerDecoder(
        TransformerDecoderLayer,
        num_layers=2,
        decoder_layer_kwargs=decoder_layer_kwargs,
        add_embedding_layer=True,
        embedding_layer_kwargs=embedding_layer_kwargs,
        device=device,
    )

    assign_acc(model, acc)

    model.layers[0].token_embedding.set_mapping(
        [
            [(2, 0, 1, 512, 512)],
            [(2, 1, 1, 512, 512)],
            [(2, 2, 1, 512, 512)],
            [(2, 3, 1, 512, 512)],
        ]
    )
    model.layers[0].pos_embedding.set_mapping([[(2, 4, 1, 512, 512)]])
    model.layers[0].k_proj_in.set_mapping([[(0, 1, 1, 512, 512)]])
    model.layers[0].q_proj_in.set_mapping([[(0, 2, 1, 512, 512)]])
    model.layers[0].v_proj_in.set_mapping([[(0, 3, 1, 512, 512)]])
    model.layers[0].out_proj.set_mapping([[(0, 4, 1, 512, 512)]])
    model.layers[0].ffn1.set_mapping([[(0, 5, 1, 512, 512)]])
    model.layers[0].ffn2.set_mapping([[(0, 6, 1, 512, 512)]])

    model.layers[0].k_proj_out.set_mapping([[(1, 1, 1, 512, 512)]])
    model.layers[0].q_proj_out.set_mapping([[(1, 2, 1, 512, 512)]])
    model.layers[0].v_proj_out.set_mapping([[(1, 3, 1, 512, 512)]])
    model.layers[1].out_proj.set_mapping([[(1, 4, 1, 512, 512)]])
    model.layers[1].ffn1.set_mapping([[(1, 5, 1, 512, 512)]])
    model.layers[1].ffn2.set_mapping([[(1, 6, 1, 512, 512)]])
    concrete_args = {"seq_length": seq_length, "memory": None}
    return model, concrete_args


def test_decoder_stack_with_embeddings_and_memory():
    acc, decoder_layer_kwargs = get_acc_and_kwargs()
    seq_length = 5
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": seq_length,
    }
    decoder_layer_kwargs["with_memory"] = True
    model = TransformerDecoder(
        TransformerDecoderLayer,
        num_layers=2,
        decoder_layer_kwargs=decoder_layer_kwargs,
        add_embedding_layer=True,
        embedding_layer_kwargs=embedding_layer_kwargs,
        device=device,
    )

    assign_acc(model, acc)

    model.layers[0].token_embedding.set_mapping(
        [
            [(2, 0, 1, 512, 512)],
            [(2, 1, 1, 512, 512)],
            [(2, 2, 1, 512, 512)],
            [(2, 3, 1, 512, 512)],
        ]
    )
    model.layers[0].pos_embedding.set_mapping([[(2, 4, 1, 512, 512)]])
    model.layers[0].k_proj_in.set_mapping([[(0, 1, 1, 512, 512)]])
    model.layers[0].q_proj_in.set_mapping([[(0, 2, 1, 512, 512)]])
    model.layers[0].v_proj_in.set_mapping([[(0, 3, 1, 512, 512)]])
    model.layers[0].out_proj.set_mapping([[(0, 4, 1, 512, 512)]])
    model.layers[0].k_proj_cross.set_mapping([[(0, 7, 1, 512, 512)]])
    model.layers[0].q_proj_cross.set_mapping([[(0, 8, 1, 512, 512)]])
    model.layers[0].v_proj_cross.set_mapping([[(0, 9, 1, 512, 512)]])
    model.layers[0].out_proj_cross.set_mapping([[(0, 10, 1, 512, 512)]])
    model.layers[0].ffn1.set_mapping([[(0, 5, 1, 512, 512)]])
    model.layers[0].ffn2.set_mapping([[(0, 6, 1, 512, 512)]])

    model.layers[0].k_proj_out.set_mapping([[(1, 1, 1, 512, 512)]])
    model.layers[0].q_proj_out.set_mapping([[(1, 2, 1, 512, 512)]])
    model.layers[0].v_proj_out.set_mapping([[(1, 3, 1, 512, 512)]])
    model.layers[1].out_proj.set_mapping([[(1, 4, 1, 512, 512)]])
    model.layers[1].k_proj_cross.set_mapping([[(1, 7, 1, 512, 512)]])
    model.layers[1].q_proj_cross.set_mapping([[(1, 8, 1, 512, 512)]])
    model.layers[1].v_proj_cross.set_mapping([[(1, 9, 1, 512, 512)]])
    model.layers[1].out_proj_cross.set_mapping([[(1, 10, 1, 512, 512)]])
    model.layers[1].ffn1.set_mapping([[(1, 5, 1, 512, 512)]])
    model.layers[1].ffn2.set_mapping([[(1, 6, 1, 512, 512)]])

    memory_len = 2 * seq_length
    memory = torch.randn((2, memory_len, 512), device=device)
    concrete_args = {
        "seq_length": seq_length,
        "memory": memory,
        "memory_len": memory_len,
    }
    return model, concrete_args


if __name__ == "__main__":
    plot_graphs = False

    available_test_funcs = [
        test_decoder_stack,
        test_decoder_stack_with_embeddings,
        test_decoder_stack_with_embeddings_and_memory,
    ]

    num_sequences = 2
    for test_func in available_test_funcs:
        model, concrete_args = test_func()

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

    if plot_graphs:
        plot_graph(symbolic_traced, "results/unpruned.svg")
        prune_and_merge_ops(graph=symbolic_traced.graph)
        plot_graph(symbolic_traced, "results/pruned.svg")
        remove_unimportant_nodes_and_reconnect(graph=symbolic_traced.graph)
        plot_graph(symbolic_traced, "results/final.svg")
