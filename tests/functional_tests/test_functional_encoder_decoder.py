import torch
from torch.nn import Transformer

from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.models import EncoderDecoderTransformer
from threedsim.modules import TransformerDecoderLayer, TransformerEncoderLayer
from threedsim.modules.base import assign_acc, set_weight
from threedsim.utils import set_seed


class TmpModel(torch.nn.Module):
    def __init__(
        self,
        vocab_size,
        max_seq_len,
        d_model,
        nhead,
        num_encoder_layer,
        num_decoder_layers,
        dim_feedforward,
        device,
    ):
        super().__init__()
        self.transformer = Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layer,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=0.0,
            norm_first=False,
            batch_first=True,
            device=device,
        )
        self.enc_token_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device
        )
        self.enc_pos_embedding = torch.nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=d_model, device=device
        )

        self.dec_token_embedding = torch.nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=d_model, device=device
        )
        self.dec_pos_embedding = torch.nn.Embedding(
            num_embeddings=max_seq_len, embedding_dim=d_model, device=device
        )

        self.lm_head = torch.nn.Linear(
            in_features=d_model, out_features=vocab_size, device=device, bias=False
        )

        self.max_seq_len = max_seq_len

    @torch.no_grad()
    def generate(self, src, max_new_tokens, max_seq_len, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        device = src.device
        pos = torch.arange(
            0, src.size(-1), dtype=torch.long, device=device
        )  # shape (t)
        src = self.enc_token_embedding(src) + self.enc_pos_embedding(pos)
        memory = self.transformer.encoder(src)
        tgt_mask = Transformer.generate_square_subsequent_mask(
            max_seq_len, device=device
        )

        # we start with the BOS token
        idx = torch.tensor(0).long().view(1, 1)

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = (
                idx if idx.size(1) <= self.max_seq_len else idx[:, -self.max_seq_len :]
            )
            # embed the current idx
            pos = torch.arange(0, idx_cond.size(-1), dtype=torch.long, device=device)
            idx_cond_embed = self.dec_token_embedding(
                idx_cond
            ) + self.dec_pos_embedding(pos)
            # forward the model to get the logits for the index in the sequence
            logits = self.transformer.decoder(
                idx_cond_embed,
                memory,
                tgt_mask[: idx_cond.size(-1), : idx_cond.size(-1)],
            )
            # pluck the logits at the final step and scale by desired temperature
            logits = self.lm_head(logits[:, -1, :] / temperature)
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            # apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx


def test_enc_dec():
    set_seed(0)
    bsz = 1
    src_len = 10

    transformer = TmpModel(
        vocab_size=2048,
        max_seq_len=128,
        d_model=512,
        nhead=8,
        num_encoder_layer=2,
        num_decoder_layers=2,
        dim_feedforward=512,
        device="cpu",
    )
    # set bias to zero
    for n, p in transformer.named_parameters():
        if "bias" in n:
            p.data *= 0
    transformer.eval()

    config = AcceleratorConfig(tiles=10, tiers=21, tier_shape=(512, 512))
    acc = Accelerator(config, device="cpu")
    enc_dec_layer_kwargs = {
        "d_model": 512,
        "nhead": 8,
        "dim_feedforward": 512,
    }
    embedding_layer_kwargs = {
        "vocab_size": 2048,
        "embedding_dim": 512,
        "max_seq_length": 128,
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
        device="cpu",
    )

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

    t_sd = transformer.state_dict()

    q, k, v = torch.split(
        t_sd["transformer.encoder.layers.0.self_attn.in_proj_weight"], 512
    )
    set_weight(
        model.encoder_stack.layers[0].token_embedding,
        t_sd["enc_token_embedding.weight"],
    )
    set_weight(
        model.encoder_stack.layers[0].pos_embedding, t_sd["enc_pos_embedding.weight"]
    )
    set_weight(model.encoder_stack.layers[0].k_proj_in, k.T)
    set_weight(model.encoder_stack.layers[0].q_proj_in, q.T)
    set_weight(model.encoder_stack.layers[0].v_proj_in, v.T)
    set_weight(
        model.encoder_stack.layers[0].out_proj,
        t_sd["transformer.encoder.layers.0.self_attn.out_proj.weight"].T,
    )
    set_weight(
        model.encoder_stack.layers[0].ffn1,
        t_sd["transformer.encoder.layers.0.linear1.weight"].T,
    )
    set_weight(
        model.encoder_stack.layers[0].ffn2,
        t_sd["transformer.encoder.layers.0.linear2.weight"].T,
    )

    q, k, v = torch.split(
        t_sd["transformer.encoder.layers.1.self_attn.in_proj_weight"], 512
    )
    set_weight(model.encoder_stack.layers[0].k_proj_out, k.T)
    set_weight(model.encoder_stack.layers[0].q_proj_out, q.T)
    set_weight(model.encoder_stack.layers[0].v_proj_out, v.T)
    set_weight(
        model.encoder_stack.layers[1].out_proj,
        t_sd["transformer.encoder.layers.1.self_attn.out_proj.weight"].T,
    )
    set_weight(
        model.encoder_stack.layers[1].ffn1,
        t_sd["transformer.encoder.layers.1.linear1.weight"].T,
    )
    set_weight(
        model.encoder_stack.layers[1].ffn2,
        t_sd["transformer.encoder.layers.1.linear2.weight"].T,
    )

    q_self, k_self, v_self = torch.split(
        t_sd["transformer.decoder.layers.0.self_attn.in_proj_weight"], 512
    )
    q_cross, k_cross, v_cross = torch.split(
        t_sd["transformer.decoder.layers.0.multihead_attn.in_proj_weight"], 512
    )
    set_weight(
        model.decoder_stack.layers[0].token_embedding,
        t_sd["dec_token_embedding.weight"],
    )
    set_weight(
        model.decoder_stack.layers[0].pos_embedding, t_sd["dec_pos_embedding.weight"]
    )

    set_weight(model.decoder_stack.layers[0].k_proj_in, k_self.T)
    set_weight(model.decoder_stack.layers[0].q_proj_in, q_self.T)
    set_weight(model.decoder_stack.layers[0].v_proj_in, v_self.T)
    set_weight(
        model.decoder_stack.layers[0].out_proj,
        t_sd["transformer.decoder.layers.0.self_attn.out_proj.weight"].T,
    )
    set_weight(model.decoder_stack.layers[0].k_proj_cross, k_cross.T)
    set_weight(model.decoder_stack.layers[0].q_proj_cross, q_cross.T)
    set_weight(model.decoder_stack.layers[0].v_proj_cross, v_cross.T)
    set_weight(
        model.decoder_stack.layers[0].out_proj_cross,
        t_sd["transformer.decoder.layers.0.multihead_attn.out_proj.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[0].ffn1,
        t_sd["transformer.decoder.layers.0.linear1.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[0].ffn2,
        t_sd["transformer.decoder.layers.0.linear2.weight"].T,
    )

    q_self, k_self, v_self = torch.split(
        t_sd["transformer.decoder.layers.1.self_attn.in_proj_weight"], 512
    )
    q_cross, k_cross, v_cross = torch.split(
        t_sd["transformer.decoder.layers.1.multihead_attn.in_proj_weight"], 512
    )
    set_weight(model.decoder_stack.layers[0].k_proj_out, k_self.T)
    set_weight(model.decoder_stack.layers[0].q_proj_out, q_self.T)
    set_weight(model.decoder_stack.layers[0].v_proj_out, v_self.T)
    set_weight(
        model.decoder_stack.layers[1].out_proj,
        t_sd["transformer.decoder.layers.1.self_attn.out_proj.weight"].T,
    )
    set_weight(model.decoder_stack.layers[1].k_proj_cross, k_cross.T)
    set_weight(model.decoder_stack.layers[1].q_proj_cross, q_cross.T)
    set_weight(model.decoder_stack.layers[1].v_proj_cross, v_cross.T)
    set_weight(
        model.decoder_stack.layers[1].out_proj_cross,
        t_sd["transformer.decoder.layers.1.multihead_attn.out_proj.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[1].ffn1,
        t_sd["transformer.decoder.layers.1.linear1.weight"].T,
    )
    set_weight(
        model.decoder_stack.layers[1].ffn2,
        t_sd["transformer.decoder.layers.1.linear2.weight"].T,
    )

    set_weight(model.lm_head, t_sd["lm_head.weight"].T)

    idx = torch.randint(low=0, high=2048, size=(bsz, src_len))
    set_seed(1)
    y_pred = model(
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
    set_seed(1)
    target = transformer.generate(idx, max_new_tokens=3, max_seq_len=128)

    print("target", target)
    print("pred", y_pred)


if __name__ == "__main__":
    test_enc_dec()
