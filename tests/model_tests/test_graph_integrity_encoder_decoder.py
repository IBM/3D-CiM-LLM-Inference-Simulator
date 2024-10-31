from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.graph import prune_and_merge_ops, remove_unimportant_nodes_and_reconnect
from threedsim.inference import trace_model
from threedsim.models import EncoderDecoderTransformer
from threedsim.modules import TransformerDecoderLayer, TransformerEncoderLayer
from threedsim.modules.base import assign_acc, fill_name_fields, make_traceable
from threedsim.plotting import plot_graph


def test_encoder_decoder():
    config = AcceleratorConfig(tiles=10, tiers=21, tier_shape=(512, 512))

    plot_graphs = False
    num_sequences = 2
    context_len = 4
    gen_start_len = 2
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

    # need to assign accelerator before mapping
    assign_acc(model, acc)
    model.encoder_stack.layers[0].token_embedding.set_mapping(
        [
            [(2, 0, 1, 512, 512)],
            [(2, 1, 1, 512, 512)],
            [(2, 2, 1, 512, 512)],
            [(2, 3, 1, 512, 512)],
        ]
    )
    model.encoder_stack.layers[0].pos_embedding.set_mapping([[(2, 4, 1, 512, 512)]])
    model.encoder_stack.layers[0].k_proj_in.set_mapping([[(0, 1, 1, 512, 512)]])
    model.encoder_stack.layers[0].q_proj_in.set_mapping([[(0, 2, 1, 512, 512)]])
    model.encoder_stack.layers[0].v_proj_in.set_mapping([[(0, 3, 1, 512, 512)]])
    model.encoder_stack.layers[0].out_proj.set_mapping([[(0, 4, 1, 512, 512)]])
    model.encoder_stack.layers[0].ffn1.set_mapping([[(0, 5, 1, 512, 512)]])
    model.encoder_stack.layers[0].ffn2.set_mapping([[(0, 6, 1, 512, 512)]])

    model.encoder_stack.layers[0].k_proj_out.set_mapping([[(1, 1, 1, 512, 512)]])
    model.encoder_stack.layers[0].q_proj_out.set_mapping([[(1, 2, 1, 512, 512)]])
    model.encoder_stack.layers[0].v_proj_out.set_mapping([[(1, 3, 1, 512, 512)]])
    model.encoder_stack.layers[1].out_proj.set_mapping([[(1, 4, 1, 512, 512)]])
    model.encoder_stack.layers[1].ffn1.set_mapping([[(1, 5, 1, 512, 512)]])
    model.encoder_stack.layers[1].ffn2.set_mapping([[(1, 6, 1, 512, 512)]])

    model.decoder_stack.layers[0].token_embedding.set_mapping(
        [
            [(3, 0, 1, 512, 512)],
            [(3, 1, 1, 512, 512)],
            [(3, 2, 1, 512, 512)],
            [(3, 3, 1, 512, 512)],
        ]
    )
    model.decoder_stack.layers[0].pos_embedding.set_mapping([[(3, 4, 1, 512, 512)]])
    model.decoder_stack.layers[0].k_proj_in.set_mapping([[(4, 1, 1, 512, 512)]])
    model.decoder_stack.layers[0].q_proj_in.set_mapping([[(4, 2, 1, 512, 512)]])
    model.decoder_stack.layers[0].v_proj_in.set_mapping([[(4, 3, 1, 512, 512)]])
    model.decoder_stack.layers[0].out_proj.set_mapping([[(4, 4, 1, 512, 512)]])
    model.decoder_stack.layers[0].k_proj_cross.set_mapping([[(4, 7, 1, 512, 512)]])
    model.decoder_stack.layers[0].q_proj_cross.set_mapping([[(4, 8, 1, 512, 512)]])
    model.decoder_stack.layers[0].v_proj_cross.set_mapping([[(4, 9, 1, 512, 512)]])
    model.decoder_stack.layers[0].out_proj_cross.set_mapping([[(4, 10, 1, 512, 512)]])
    model.decoder_stack.layers[0].ffn1.set_mapping([[(4, 5, 1, 512, 512)]])
    model.decoder_stack.layers[0].ffn2.set_mapping([[(4, 6, 1, 512, 512)]])

    model.decoder_stack.layers[0].k_proj_out.set_mapping([[(5, 1, 1, 512, 512)]])
    model.decoder_stack.layers[0].q_proj_out.set_mapping([[(5, 2, 1, 512, 512)]])
    model.decoder_stack.layers[0].v_proj_out.set_mapping([[(5, 3, 1, 512, 512)]])
    model.decoder_stack.layers[1].out_proj.set_mapping([[(5, 4, 1, 512, 512)]])
    model.decoder_stack.layers[1].k_proj_cross.set_mapping([[(5, 7, 1, 512, 512)]])
    model.decoder_stack.layers[1].q_proj_cross.set_mapping([[(5, 8, 1, 512, 512)]])
    model.decoder_stack.layers[1].v_proj_cross.set_mapping([[(5, 9, 1, 512, 512)]])
    model.decoder_stack.layers[1].out_proj_cross.set_mapping([[(5, 10, 1, 512, 512)]])
    model.decoder_stack.layers[1].ffn1.set_mapping([[(5, 5, 1, 512, 512)]])
    model.decoder_stack.layers[1].ffn2.set_mapping([[(5, 6, 1, 512, 512)]])
    model.lm_head.set_mapping(
        [
            [
                (6, 0, 1, 512, 512),
                (6, 1, 1, 512, 512),
                (6, 2, 1, 512, 512),
                (6, 3, 1, 512, 512),
            ]
        ]
    )

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

    if plot_graphs:
        plot_graph(symbolic_traced, "results/unpruned.svg")
    prune_and_merge_ops(graph=symbolic_traced.graph)
    if plot_graphs:
        plot_graph(symbolic_traced, "results/pruned.svg")
    remove_unimportant_nodes_and_reconnect(graph=symbolic_traced.graph)
    if plot_graphs:
        plot_graph(symbolic_traced, "results/final.svg")

    graph = symbolic_traced.graph
    nodes = [*graph.nodes]
    for n in nodes:
        if n.op == "output":
            continue
        if n.kwargs["op_info"]["decoding_id"] == "memory":
            # go through the users
            for user in n.users:
                if user.kwargs["op_info"]["decoding_id"] != "memory":
                    # this connection must be from a layer norm to a cross proj
                    # print(user.kwargs["op_info"])
                    assert (
                        "layer_norm" in n.name
                        and "cross" in user.kwargs["op_info"]["layer_name"]
                    ), "conn from encoder part must feed into cross attention linear layer"


if __name__ == "__main__":
    test_encoder_decoder()
