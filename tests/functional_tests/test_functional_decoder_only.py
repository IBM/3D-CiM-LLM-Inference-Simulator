import torch
from decoder_only_model import GPT, GPTConfig

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.models import DecoderOnlyTransformer
from threedsim.modules import TransformerDecoderLayer
from threedsim.modules.base import assign_acc, set_weight
from threedsim.utils import set_seed


def test_gpt2():
    set_seed(0)

    # gpt2 from nanoGPT
    config = GPTConfig(
        block_size=128,
        vocab_size=2048,
        n_layer=2,
        n_head=8,
        n_embd=512,
        n_ff=512,
        bias=False,
    )

    bsz = 1
    seq_len = 10
    gpt_model = GPT(config=config)
    gpt_model.eval()

    decoder_layer_kwargs = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 512,
        "norm_first": True,
    }
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": 128,
    }
    model = DecoderOnlyTransformer(
        TransformerDecoderLayer,
        num_layers=2,
        decoder_layer_kwargs=decoder_layer_kwargs,
        embedding_layer_kwargs=embedding_layer_kwargs,
        device="cpu",
    )
    model.eval()

    # assign the weights
    state_dict_gpt = gpt_model.state_dict()

    config = AcceleratorConfig(tiles=4, tiers=21, tier_shape=(512, 512))
    acc = Accelerator(config, device="cpu")
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

    # need to transpose since torch performs x W.T
    # don't do this for embeddings.
    q, k, v = state_dict_gpt["transformer.h.0.attn.c_attn.weight"].split(512)
    set_weight(
        model.decoder_stack.layers[0].token_embedding,
        state_dict_gpt["transformer.wte.weight"],
    )
    set_weight(
        model.decoder_stack.layers[0].pos_embedding,
        state_dict_gpt["transformer.wpe.weight"],
    )
    set_weight(model.decoder_stack.layers[0].k_proj_in, k.T)
    set_weight(model.decoder_stack.layers[0].q_proj_in, q.T)
    set_weight(model.decoder_stack.layers[0].v_proj_in, v.T)
    set_weight(
        model.decoder_stack.layers[0].out_proj,
        state_dict_gpt["transformer.h.0.attn.c_proj.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[0].ffn1,
        state_dict_gpt["transformer.h.0.mlp.c_fc.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[0].ffn2,
        state_dict_gpt["transformer.h.0.mlp.c_proj.weight"].T,
    )

    q, k, v = state_dict_gpt["transformer.h.1.attn.c_attn.weight"].split(512)
    set_weight(model.decoder_stack.layers[0].k_proj_out, k.T)
    set_weight(model.decoder_stack.layers[0].q_proj_out, q.T)
    set_weight(model.decoder_stack.layers[0].v_proj_out, v.T)
    set_weight(
        model.decoder_stack.layers[1].out_proj,
        state_dict_gpt["transformer.h.1.attn.c_proj.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[1].ffn1,
        state_dict_gpt["transformer.h.1.mlp.c_fc.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[1].ffn2,
        state_dict_gpt["transformer.h.1.mlp.c_proj.weight"].T,
    )

    set_weight(model.lm_head, state_dict_gpt["lm_head.weight"].T)

    idx = torch.randint(low=0, high=2048, size=(bsz, seq_len))
    set_seed(0)
    y_pred = model(src=idx.squeeze(), start_len=10, target_len=12)
    set_seed(0)
    target = gpt_model.generate(idx=idx, max_new_tokens=2).squeeze()

    assert torch.allclose(y_pred, target)


if __name__ == "__main__":
    test_gpt2()
