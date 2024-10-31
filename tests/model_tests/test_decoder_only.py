from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.graph import prune_and_merge_ops, remove_unimportant_nodes_and_reconnect
from threedsim.inference import schedule_execution, trace_model
from threedsim.models import DecoderOnlyTransformer
from threedsim.modules import TransformerDecoderLayer
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable
from threedsim.plotting import plot_graph


def test_decoder_only():
    config = AcceleratorConfig(tiles=4, tiers=21, tier_shape=(512, 512))

    plot_graphs = False
    num_sequences = 2
    start_len = 2
    target_len = 4

    device = "meta"
    acc = Accelerator(config, device=device)
    decoder_layer_kwargs = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 512,
    }
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": target_len,
    }
    model = DecoderOnlyTransformer(
        TransformerDecoderLayer,
        num_layers=2,
        decoder_layer_kwargs=decoder_layer_kwargs,
        embedding_layer_kwargs=embedding_layer_kwargs,
        device=device,
    )

    # need to assign accelerator before mapping
    assign_acc(model, acc)

    model.decoder_stack.layers[0].token_embedding.set_mapping(
        [
            [(2, 0, 1, 512, 512)],
            [(2, 1, 1, 512, 512)],
            [(2, 2, 1, 512, 512)],
            [(2, 3, 1, 512, 512)],
        ]
    )
    model.decoder_stack.layers[0].pos_embedding.set_mapping([[(2, 4, 1, 512, 512)]])
    model.decoder_stack.layers[0].k_proj_in.set_mapping([[(0, 1, 1, 512, 512)]])
    model.decoder_stack.layers[0].q_proj_in.set_mapping([[(0, 2, 1, 512, 512)]])
    model.decoder_stack.layers[0].v_proj_in.set_mapping([[(0, 3, 1, 512, 512)]])
    model.decoder_stack.layers[0].out_proj.set_mapping([[(0, 4, 1, 512, 512)]])
    model.decoder_stack.layers[0].ffn1.set_mapping([[(0, 5, 1, 512, 512)]])
    model.decoder_stack.layers[0].ffn2.set_mapping([[(0, 6, 1, 512, 512)]])

    model.decoder_stack.layers[0].k_proj_out.set_mapping([[(1, 1, 1, 512, 512)]])
    model.decoder_stack.layers[0].q_proj_out.set_mapping([[(1, 2, 1, 512, 512)]])
    model.decoder_stack.layers[0].v_proj_out.set_mapping([[(1, 3, 1, 512, 512)]])
    model.decoder_stack.layers[1].out_proj.set_mapping([[(1, 4, 1, 512, 512)]])
    model.decoder_stack.layers[1].ffn1.set_mapping([[(1, 5, 1, 512, 512)]])
    model.decoder_stack.layers[1].ffn2.set_mapping([[(1, 6, 1, 512, 512)]])
    model.lm_head.set_mapping(
        [
            [
                (3, 0, 1, 512, 512),
                (3, 1, 1, 512, 512),
                (3, 2, 1, 512, 512),
                (3, 3, 1, 512, 512),
            ]
        ]
    )

    fill_name_fields(model)
    make_traceable(model, is_traceable=True)

    symbolic_traced = trace_model(
        model,
        num_sequences,
        concrete_args={"start_len": start_len, "target_len": target_len},
    )
    execution_time, memory, peak_memory, energy, flops, _, _ = schedule_execution(
        symbolic_traced.graph,
        accelerator=model.accelerator,
        plot=True,
        plot_dir="results",
    )
    print(f"Execution took {execution_time} units")
    print(f"Required {peak_memory} bytes of scratchpad memory")
    print(f"Spent {energy} xJ of energy")

    # Print tile efficiency
    tile_eff = [(tile.active_time / execution_time) * 100 for tile in acc.tiles]
    print("Tile efficiencies:")
    for i, eff in enumerate(tile_eff):
        print(f"\tTile {i} -> {eff:.2f}%")

    if plot_graphs:
        plot_graph(symbolic_traced, "results/unpruned.svg")
        prune_and_merge_ops(graph=symbolic_traced.graph)
        plot_graph(symbolic_traced, "results/pruned.svg")
        remove_unimportant_nodes_and_reconnect(graph=symbolic_traced.graph)
        plot_graph(symbolic_traced, "results/final.svg")


if __name__ == "__main__":
    test_decoder_only()
