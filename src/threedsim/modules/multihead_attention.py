import math

import torch
import torch.fx as fx

from .base import BaseModule


class MultiheadAttention(BaseModule, torch.nn.Module):
    def __init__(self, d_model):
        super().__init__()
        torch._C._log_api_usage_once(f"torch.nn.modules.{self.__class__.__name__}")
        self.d_model = d_model

    def forward(
        self,
        q,
        k,
        v,
        seq_length,
        attn_mask,
        two_stage: bool = False,
        op_info: dict = None,
    ):
        mha_func = _mha if not two_stage else _mha_sep
        att_output = mha_func(
            q, k, v, seq_length, self.d_model, attn_mask, op_info=op_info
        )
        return att_output


@fx.wrap
def _mha(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_length: int,
    d_model: int,
    attn_mask,
    op_info: dict = None,
):
    # manual implementation of attention
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if attn_mask is not None:
        att = att.masked_fill(
            attn_mask[:, :, :seq_length, :seq_length] == 0, float("-inf")
        )
    att = torch.nn.functional.softmax(att, dim=-1)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    y = (
        y.transpose(1, 2).contiguous().view(seq_length, d_model)
    )  # re-assemble all head outputs side by side
    return y


@fx.wrap
def qk_aux(
    q: torch.Tensor,
    k: torch.Tensor,
    seq_length: int,
    attn_mask: torch.Tensor,
    op_info: dict = None,
):
    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
    if attn_mask is not None:
        att = att.masked_fill(
            attn_mask[:, :, :seq_length, :seq_length] == 0, float("-inf")
        )
    att = torch.nn.functional.softmax(att, dim=-1)
    return att


@fx.wrap
def pv_aux(
    attn_weight: torch.Tensor,
    v: torch.Tensor,
    seq_length: int,
    d_model: int,
    op_info: dict = None,
):
    attn_output = attn_weight @ v
    attn_output = (
        attn_output.permute(2, 0, 1, 3).contiguous().view(1 * seq_length, d_model)
    )
    return attn_output


def _mha_sep(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_length: int,
    d_model: int,
    attn_mask,
    op_info: dict = None,
):
    attn_weight = qk_aux(q, k, seq_length, attn_mask, op_info=op_info)
    attn_output = pv_aux(attn_weight, v, seq_length, d_model, op_info=op_info)
    return attn_output
