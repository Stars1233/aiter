# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2025, Advanced Micro Devices, Inc. All rights reserved.

import torch
import aiter
from aiter import dtypes
from aiter.test_mha_common import (
    attention_ref,
    attn_bias_from_alibi_slopes,
    ck_randval_to_dropout_mask,
    convert_flash_attn_S_to_softmax,
    generate_qkv,
    generate_random_padding_mask,
    pad_rearrange_dropout_mask_hts_to_bhss,
)
import pytest
import argparse


def run_torch(
    q,
    k,
    v,
    query_padding_mask,
    key_padding_mask,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    upcast=True,
    reorder_ops=False,
):
    (b, seqlen_q, _, _) = q.shape
    (_, seqlen_k, _, _) = k.shape

    if bias is not None:
        attn_bias = bias.reshape(b, 1, seqlen_q, seqlen_k)
    elif alibi_slopes is not None:
        attn_bias = attn_bias_from_alibi_slopes(
            alibi_slopes,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            causal=causal,
        )
    else:
        attn_bias = None

    out, _ = attention_ref(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        dropout_p,
        dropout_mask,
        causal=causal,
        window_size=window_size,
        upcast=upcast,
        reorder_ops=reorder_ops,
    )

    if dout == None:
        return out
    else:
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), dout)
        return out, dq, dk, dv


def run_ck(
    q,
    k,
    v,
    query_padding_mask,
    key_padding_mask,
    min_seqlen_q=0,
    bias=None,
    alibi_slopes=None,
    dout=None,
    dropout_p=0.0,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    deterministic=False,
    return_lse=False,
    return_attn_probs=False,
):
    (
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        q,
        k,
        v,
        output_pad_fn,
        dq_pad_fn,
        dk_pad_fn,
    ) = generate_qkv(q, k, v, query_padding_mask, key_padding_mask, kvpacked=False)
    if bias is not None:
        # TODO - implement generate_bias() to unpad
        total_q = q_unpad.shape[0]
        assert total_q == batch_size * max_seqlen_q
        assert q.shape[1] == max_seqlen_q
        assert k.shape[1] == max_seqlen_k
        bias_unpad = bias.reshape(batch_size * max_seqlen_q, max_seqlen_k)
    else:
        bias_unpad = None

    outputs = aiter.flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        min_seqlen_q=min_seqlen_q,
        dropout_p=dropout_p,
        causal=causal,
        window_size=window_size,
        bias=bias_unpad,
        alibi_slopes=alibi_slopes,
        deterministic=deterministic,
        return_lse=return_lse,
        return_attn_probs=return_attn_probs,
    )

    if type(outputs) is tuple:
        out = output_pad_fn(outputs[0])
    else:
        out = output_pad_fn(outputs)

    if dropout_p > 0.0 and return_attn_probs:
        (_, seqlen_q, _, d) = q.shape
        (_, seqlen_k, _, d) = k.shape
        S_dmask = outputs[-1]
        S_dmask = ck_randval_to_dropout_mask(S_dmask, dropout_p)
        S_dmask = pad_rearrange_dropout_mask_hts_to_bhss(
            S_dmask, cu_seqlens_q, seqlen_q, seqlen_k
        )
        S_dmask_converted = convert_flash_attn_S_to_softmax(
            S_dmask,
            seqlen_q,
            seqlen_k,
            query_padding_mask,
            key_padding_mask,
            d,
            dropout_p > 0.0,
            causal=causal,
            window_size=window_size,
        )
        dropout_mask = S_dmask_converted >= 0
    else:
        dropout_mask = None

    if dout is None or not return_lse:
        return out, dropout_mask, None, None, None
    else:
        dq_unpad, dk_unpad, dv_unpad = torch.autograd.grad(
            out, (q_unpad, k_unpad, v_unpad), dout
        )
        dq = dq_pad_fn(dq_unpad)
        dk = dk_pad_fn(dk_unpad)
        dv = dk_pad_fn(dv_unpad)
        return out, dropout_mask, dq, dk, dv


@pytest.mark.parametrize("dtype", [dtypes.fp16, dtypes.bf16])
@pytest.mark.parametrize("mha_type", ["mha", "mqa", "gqa"])
@pytest.mark.parametrize("deterministic", [True, False])
@pytest.mark.parametrize("bias_type", ["no", "alibi"])
@pytest.mark.parametrize("local", [False, True])
@pytest.mark.parametrize("causal", [False, True])
@pytest.mark.parametrize("min_seqlen_q", [0])
@pytest.mark.parametrize("dropout_p", [0.0, 0.17])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("return_lse", [False, True])
@pytest.mark.parametrize("return_attn_probs", [False, True])
@pytest.mark.parametrize("nheads", [9])
@pytest.mark.parametrize(
    "d,d_v",
    [
        (32, 32),
        (40, 40),
        (59, 59),
        (64, 64),
        (96, 96),
        (111, 111),
        (128, 128),
        (160, 160),
        (192, 192),
        (224, 224),
        (256, 256),
    ],
)
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 147),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_flash_attn_varlen_func(
    batch_size,
    nheads,
    seqlen_q,
    seqlen_k,
    d,
    d_v,
    min_seqlen_q,
    dropout_p,
    causal,
    local,
    bias_type,
    deterministic,
    mha_type,
    dtype,
    return_lse,
    return_attn_probs,
):
    torch.random.manual_seed(0)
    nheads_k = nheads if mha_type == "mha" else (1 if mha_type == "mqa" else 3)
    assert nheads % nheads_k == 0
    window_size = (-1, -1) if not local else torch.randint(0, seqlen_k, (2,))

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device="cuda", dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    v = torch.randn(
        batch_size,
        seqlen_k,
        nheads_k,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    if bias_type == "bias":
        # TODO - We need to implement unpad bias [batch_size, seqlen_q, seqlen_k] -> [total_q, max_seqlen_k]
        # Let total_q = batch_size * seqlen_q to pass the test for now
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, "cuda", mode="full"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, "cuda", mode="full"
        )
    else:
        query_padding_mask = generate_random_padding_mask(
            seqlen_q, batch_size, "cuda", mode="random"
        )
        key_padding_mask = generate_random_padding_mask(
            seqlen_k, batch_size, "cuda", mode="random"
        )

    attn_bias = None
    alibi_slopes = None
    if bias_type == "bias":
        attn_bias = torch.randn(
            batch_size,
            seqlen_q,
            seqlen_k,
            device="cuda",
            dtype=dtype,
            requires_grad=True,
        )
    elif bias_type == "alibi":
        alibi_slopes = torch.rand(batch_size, nheads, device="cuda", dtype=dtypes.fp32)

    dout = torch.randn(
        batch_size,
        seqlen_q,
        nheads,
        d_v,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )

    if dropout_p > 0:
        return_attn_probs = True

    out, dropout_mask, dq, dk, dv = run_ck(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        min_seqlen_q,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        causal,
        window_size,
        deterministic,
        return_lse,
        return_attn_probs,
    )

    out_ref, dq_ref, dk_ref, dv_ref = run_torch(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
    )

    out_pt, dq_pt, dk_pt, dv_pt = run_torch(
        q,
        k,
        v,
        query_padding_mask,
        key_padding_mask,
        attn_bias,
        alibi_slopes,
        dout,
        dropout_p,
        dropout_mask,
        causal,
        window_size,
        upcast=False,
        reorder_ops=True,
    )

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    out_tol = max(4 * (out_pt - out_ref).abs().max().item(), 0.01)
    # assert (out - out_ref).abs().max().item() <= out_tol

    # TODO: Support varlen bwd for bias
    if bias_type == "bias":
        pytest.skip("Does not support varlen bwd for bias")

    if dq is not None:
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")

        dq_tol = max(10 * (dq_pt - dq_ref).abs().max().item(), 0.01)
        dk_tol = max(10 * (dk_pt - dk_ref).abs().max().item(), 0.01)
        dv_tol = max(10 * (dv_pt - dv_ref).abs().max().item(), 0.01)

        assert (dq - dq_ref).abs().max().item() <= dq_tol
        assert (dk - dk_ref).abs().max().item() <= dk_tol
        assert (dv - dv_ref).abs().max().item() <= dv_tol


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        nargs="?",
        default=4,
        help="""Batch size.
    e.g.: -b 16""",
    )
    parser.add_argument(
        "-nh",
        "--nheads",
        type=int,
        nargs="?",
        default=4,
        help="""Number of attention heads.
    e.g. -nh 4""",
    )
    parser.add_argument(
        "-s",
        "--seqlen_q_k",
        type=dtypes.str2tuple,
        nargs="?",
        default=(4, 4),
        help="""Sequence length of query&key.
    e.g. -s 4,4""",
    )
    parser.add_argument(
        "-d",
        type=int,
        nargs="?",
        default=192,
        help="""Dimension of query&key.
    e.g. -d 192""",
    )
    parser.add_argument(
        "-dv",
        type=int,
        nargs="?",
        default=192,
        help="""Dimension of value.
    e.g. -dv 192""",
    )
    parser.add_argument(
        "-dp",
        "--dropout_p",
        type=float,
        nargs="?",
        default=0.0,
        help="""Dropout probability."
    e.g. -dp 0.0""",
    )
    parser.add_argument(
        "-msq",
        "--min_seqlen_q",
        type=int,
        nargs="?",
        default=1,
        help="""Minimum sequence length of query.
    e.g. -msq 1""",
    )
    parser.add_argument(
        "-c",
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Causal attention, default is True.
    -c or --causal    # enable causal attention
    --no-causal       # disable causal attention""",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        default=False,
        help="""Local attention. default is False.
    e.g. -l or --local    # enable local attention""",
    )
    parser.add_argument(
        "-bt",
        "--bias_type",
        type=str,
        default="no",
        help="Type of bias.",
    )
    parser.add_argument(
        "-det",
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="""Deterministic attention, default is True.
    -det or --deterministic    # enable deterministic attention
    --no-deterministic         # disable deterministic attention""",
    )
    parser.add_argument(
        "-mha",
        "--mha_type",
        type=str,
        default="mha",
        help="""Type of multi-head attention.
    e.g. -mha mha/mqa/gqa""",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        type=str,
        default="bf16",
        help="""Data type.
    e.g.: -dt bf16""",
    )
    parser.add_argument(
        "-rlse",
        "--return_lse",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""return logsumexp, default is False.
    -rlse or --return_lse    # enable return logsumexp
    --no-return_lse          # disable return logsumexp""",
    )
    parser.add_argument(
        "-rap",
        "--return_attn_probs",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="""return attention probabilities, default is False.
    -rap or --return_attn_probs    # enable return attention probabilities
    --no-return_attn_probs        # disable return attention probabilities""",
    )

    args = parser.parse_args()
    dtype = dtypes.d_dtypes[args.dtype]
    (seqlen_q, seqlen_k) = args.seqlen_q_k

    test_flash_attn_varlen_func(
        args.batch_size,
        args.nheads,
        seqlen_q,
        seqlen_k,
        args.d,
        args.dv,
        args.min_seqlen_q,
        args.dropout_p,
        args.causal,
        args.local,
        args.bias_type,
        args.deterministic,
        args.mha_type,
        dtype,
        args.return_lse,
        args.return_attn_probs,
    )
