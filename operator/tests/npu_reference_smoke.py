#!/usr/bin/env python3
"""Randomized paged-KV correctness smoke for the bounded AIV kernel."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch_npu  # noqa: F401


HEADS = 32
KV_HEADS = 8
HEAD_DIM = 128
PAGE_SIZE = 128
SINK = 8
WINDOW = 512
DENSE_TAIL = 128


def selected_tokens(
    query_position: int,
    seq_len: int,
    prompt_len: int,
) -> list[int]:
    causal_end = min(query_position + 1, seq_len)
    sparse_begin = SINK + WINDOW + 1
    sparse_end = max(0, prompt_len - DENSE_TAIL)
    if not (sparse_begin <= query_position < sparse_end):
        return list(range(causal_end))

    sink_end = min(SINK, causal_end)
    local_begin = max(0, query_position - WINDOW)
    if local_begin <= sink_end:
        return list(range(causal_end))
    return [*range(sink_end), *range(local_begin, causal_end)]


def cpu_reference(
    query_bf16: torch.Tensor,
    key_cache_bf16: torch.Tensor,
    value_cache_bf16: torch.Tensor,
    block_table: torch.Tensor,
    query_start: int,
    seq_len: int,
    prompt_len: int,
    scale: float,
) -> torch.Tensor:
    query = query_bf16.float()
    key_cache = key_cache_bf16.float()
    value_cache = value_cache_bf16.float()
    table = block_table[0].tolist()
    result = torch.empty_like(query)

    for row in range(query.size(0)):
        query_position = query_start + row
        tokens = selected_tokens(query_position, seq_len, prompt_len)
        for query_head in range(HEADS):
            kv_head = query_head // (HEADS // KV_HEADS)
            logical_key = torch.stack(
                [
                    key_cache[
                        table[token // PAGE_SIZE],
                        token % PAGE_SIZE,
                        kv_head,
                    ]
                    for token in tokens
                ]
            )
            logical_value = torch.stack(
                [
                    value_cache[
                        table[token // PAGE_SIZE],
                        token % PAGE_SIZE,
                        kv_head,
                    ]
                    for token in tokens
                ]
            )
            logits = torch.mv(logical_key, query[row, query_head]) * scale
            weights = torch.softmax(logits, dim=0)
            result[row, query_head] = torch.mv(
                logical_value.transpose(0, 1), weights
            )
    return result


def make_inputs(
    query_start: int,
    query_tokens: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    seq_len = query_start + query_tokens
    logical_pages = math.ceil(seq_len / PAGE_SIZE)
    physical_pages = logical_pages + 3
    permutation = torch.randperm(physical_pages, generator=generator)
    block_table = permutation[:logical_pages].to(torch.int32).view(1, -1)

    query = (
        torch.randn(
            query_tokens,
            HEADS,
            HEAD_DIM,
            generator=generator,
        )
        .mul_(0.5)
        .to(torch.bfloat16)
    )
    key = (
        torch.randn(
            physical_pages,
            PAGE_SIZE,
            KV_HEADS,
            HEAD_DIM,
            generator=generator,
        )
        .mul_(0.5)
        .to(torch.bfloat16)
    )
    value = (
        torch.randn(
            physical_pages,
            PAGE_SIZE,
            KV_HEADS,
            HEAD_DIM,
            generator=generator,
        )
        .mul_(0.5)
        .to(torch.bfloat16)
    )
    return query, key, value, block_table


def run_case(
    name: str,
    query_start: int,
    query_tokens: int,
    prompt_len: int,
    seed: int,
) -> dict[str, float | str | int]:
    query, key, value, block_table = make_inputs(
        query_start, query_tokens, seed
    )
    seq_len = query_start + query_tokens
    scale = HEAD_DIM**-0.5
    expected = cpu_reference(
        query,
        key,
        value,
        block_table,
        query_start,
        seq_len,
        prompt_len,
        scale,
    )

    actual_npu = torch.ops.trianglemix_reference.triangle_paged_sparse_attention(
        query.npu(),
        key.npu(),
        value.npu(),
        block_table.npu(),
        query_start,
        seq_len,
        prompt_len,
        scale,
    )
    torch.npu.synchronize()
    actual = actual_npu.cpu().float()
    expected_finite = bool(torch.isfinite(expected).all())
    actual_finite = bool(torch.isfinite(actual).all())
    if not expected_finite or not actual_finite:
        raise AssertionError(
            f"{name}: non-finite values "
            f"expected_finite={expected_finite}, "
            f"actual_finite={actual_finite}"
        )
    difference = (actual - expected).abs()
    max_abs = float(difference.max())
    mean_abs = float(difference.mean())
    max_rel = float(
        (difference / expected.abs().clamp_min(1.0e-3)).max()
    )

    if max_abs > 0.04 or mean_abs > 0.004:
        raise AssertionError(
            f"{name}: BF16 mismatch max_abs={max_abs:.6f}, "
            f"mean_abs={mean_abs:.6f}, max_rel={max_rel:.6f}"
        )
    return {
        "name": name,
        "query_start": query_start,
        "query_tokens": query_tokens,
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "finite": actual_finite,
        "max_abs": max_abs,
        "mean_abs": mean_abs,
        "max_rel": max_rel,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to triangle_paged_attention_torch*.so",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.ops.load_library(str(args.adapter.resolve()))
    cases = [
        ("dense_prefix", 0, 4, 1024, 1001),
        ("split_sink_local", 520, 4, 1024, 1002),
        ("dense_tail_boundary", 894, 4, 1024, 1003),
    ]
    for case in cases:
        print(run_case(*case), flush=True)
    print("NPU_REFERENCE_SMOKE_PASS", flush=True)


if __name__ == "__main__":
    main()
