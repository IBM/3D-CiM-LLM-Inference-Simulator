import copy

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.graph import prune_and_merge_ops, remove_unimportant_nodes_and_reconnect
from threedsim.inference import schedule_execution, trace_model
from threedsim.modules import TransformerEncoder, TransformerEncoderLayer
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable
from threedsim.plotting import plot_graph

device = "meta"


def get_acc_and_kwargs():
    config = AcceleratorConfig(tiles=10, tiers=64, tier_shape=(512, 512))
    acc = Accelerator(config, device=device)
    encoder_layer_kwargs = {
        "d_model": 784,
        "nhead": 8,
        "dim_feedforward": 1024,
    }
    return acc, encoder_layer_kwargs


def test_encoder_layer():
    acc, encoder_layer_kwargs = get_acc_and_kwargs()
    seq_length = 5
    model = TransformerEncoderLayer(
        **encoder_layer_kwargs,
        is_first=True,
        is_last=True,
        device=device,
    )
    assign_acc(model, acc)

    model.k_proj_in.set_mapping([[(0, 1, 1, 512, 512)]])
    model.q_proj_in.set_mapping([[(0, 2, 1, 512, 512)]])
    model.v_proj_in.set_mapping([[(0, 3, 1, 512, 512)]])
    model.out_proj.set_mapping([[(0, 4, 1, 512, 512)]])
    model.ffn1.set_mapping([[(0, 5, 1, 512, 512)]])
    model.ffn2.set_mapping([[(0, 6, 1, 512, 512)]])
    concrete_args = {"seq_length": seq_length}
    return model, concrete_args


def test_encoder_layer_with_embeddings():
    acc, encoder_layer_kwargs = get_acc_and_kwargs()
    seq_length = 5
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": seq_length,
    }
    model = TransformerEncoderLayer(
        **encoder_layer_kwargs,
        is_first=True,
        is_last=True,
        device=device,
        do_embedding=True,
        embedding_layer_kwargs=embedding_layer_kwargs,
    )
    assign_acc(model, acc)

    model.token_embedding.set_mapping(
        [
            [(1, 0, 1, 512, 512)],
            [(1, 1, 1, 512, 512)],
            [(1, 2, 1, 512, 512)],
            [(1, 3, 1, 512, 512)],
        ]
    )
    model.pos_embedding.set_mapping([[(1, 4, 1, 512, 512)]])
    model.k_proj_in.set_mapping([[(0, 1, 1, 512, 512)]])
    model.q_proj_in.set_mapping([[(0, 2, 1, 512, 512)]])
    model.v_proj_in.set_mapping([[(0, 3, 1, 512, 512)]])
    model.out_proj.set_mapping([[(0, 4, 1, 512, 512)]])
    model.ffn1.set_mapping([[(0, 5, 1, 512, 512)]])
    model.ffn2.set_mapping([[(0, 6, 1, 512, 512)]])
    concrete_args = {"seq_length": seq_length}
    return model, concrete_args


def test_encoder_stack():
    acc, encoder_layer_kwargs = get_acc_and_kwargs()
    seq_length = 5

    model = TransformerEncoder(
        TransformerEncoderLayer,
        num_layers=2,
        encoder_layer_kwargs=encoder_layer_kwargs,
        device=device,
    )

    assign_acc(model, acc)

    model.layers[0].k_proj_in.set_mapping(
        [
            [(0, 1, 1, 512, 512), (0, 2, 1, 512, 512)],
            [(0, 3, 1, 512, 512), (0, 4, 1, 512, 512)],
        ]
    )
    model.layers[0].q_proj_in.set_mapping(
        [
            [(0, 5, 1, 512, 512), (0, 6, 1, 512, 512)],
            [(0, 7, 1, 512, 512), (0, 8, 1, 512, 512)],
        ]
    )
    model.layers[0].v_proj_in.set_mapping(
        [
            [(0, 9, 1, 512, 512), (0, 10, 1, 512, 512)],
            [(0, 11, 1, 512, 512), (0, 12, 1, 512, 512)],
        ]
    )
    model.layers[0].out_proj.set_mapping(
        [
            [(0, 13, 1, 512, 512), (0, 14, 1, 512, 512)],
            [(0, 15, 1, 512, 512), (0, 16, 1, 512, 512)],
        ]
    )
    model.layers[0].ffn1.set_mapping(
        [
            [(0, 17, 1, 512, 512), (0, 18, 1, 512, 512)],
            [(0, 19, 1, 512, 512), (0, 20, 1, 512, 512)],
        ]
    )
    model.layers[0].ffn2.set_mapping(
        [
            [(0, 21, 1, 512, 512), (0, 22, 1, 512, 512)],
            [(0, 23, 1, 512, 512), (0, 24, 1, 512, 512)],
        ]
    )

    model.layers[0].k_proj_out.set_mapping(
        [
            [(1, 1, 1, 512, 512), (1, 2, 1, 512, 512)],
            [(1, 3, 1, 512, 512), (1, 4, 1, 512, 512)],
        ]
    )
    model.layers[0].q_proj_out.set_mapping(
        [
            [(1, 5, 1, 512, 512), (1, 6, 1, 512, 512)],
            [(1, 7, 1, 512, 512), (1, 8, 1, 512, 512)],
        ]
    )
    model.layers[0].v_proj_out.set_mapping(
        [
            [(1, 9, 1, 512, 512), (1, 10, 1, 512, 512)],
            [(1, 11, 1, 512, 512), (1, 12, 1, 512, 512)],
        ]
    )
    model.layers[1].out_proj.set_mapping(
        [
            [(1, 13, 1, 512, 512), (1, 14, 1, 512, 512)],
            [(1, 15, 1, 512, 512), (1, 16, 1, 512, 512)],
        ]
    )
    model.layers[1].ffn1.set_mapping(
        [
            [(1, 17, 1, 512, 512), (1, 18, 1, 512, 512)],
            [(1, 19, 1, 512, 512), (1, 20, 1, 512, 512)],
        ]
    )
    model.layers[1].ffn2.set_mapping(
        [
            [(1, 21, 1, 512, 512), (1, 22, 1, 512, 512)],
            [(1, 23, 1, 512, 512), (1, 24, 1, 512, 512)],
        ]
    )
    concrete_args = {"seq_length": seq_length}
    return model, concrete_args


def test_encoder_stack_with_embeddings():
    acc, encoder_layer_kwargs = get_acc_and_kwargs()
    seq_length = 5
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": seq_length,
    }
    model = TransformerEncoder(
        TransformerEncoderLayer,
        num_layers=2,
        encoder_layer_kwargs=encoder_layer_kwargs,
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
    concrete_args = {"seq_length": seq_length}
    return model, concrete_args


if __name__ == "__main__":
    plot_graphs = False
    available_test_funcs = [
        test_encoder_layer,
        test_encoder_layer_with_embeddings,
        test_encoder_stack,
        # test_encoder_stack_with_embeddings,
    ]

    num_sequences = 1
    for test_func in available_test_funcs:
        model, concrete_args = test_func()

    fill_name_fields(model)
    make_traceable(model, is_traceable=True)
    symbolic_traced = trace_model(model, num_sequences, concrete_args=concrete_args)
    if plot_graphs:
        symbolic_traced_plot = copy.deepcopy(symbolic_traced)
        plot_graph(symbolic_traced_plot, "results/unpruned.svg")
        prune_and_merge_ops(graph=symbolic_traced_plot.graph)
        plot_graph(symbolic_traced_plot, "results/pruned.svg")
        remove_unimportant_nodes_and_reconnect(graph=symbolic_traced_plot.graph)
        plot_graph(symbolic_traced_plot, "results/final.svg")

    execution_time, _, _, _, _, _, _ = schedule_execution(
        symbolic_traced.graph,
        accelerator=model.accelerator,
        plot=True,
        plot_dir="results",
    )
    print(f"Execution took {execution_time} units")
