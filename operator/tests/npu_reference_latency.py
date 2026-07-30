#!/usr/bin/env python3
"""Event timing for the correctness-only AIV reference kernel."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from npu_reference_smoke import HEAD_DIM, make_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    return parser.parse_args()


def time_case(
    name: str,
    query_start: int,
    query_tokens: int,
    prompt_len: int,
    seed: int,
    warmup: int,
    iterations: int,
    repeats: int,
) -> dict[str, object]:
    query, key, value, block_table = make_inputs(
        query_start, query_tokens, seed
    )
    seq_len = query_start + query_tokens
    scale = HEAD_DIM**-0.5
    args = (
        query.npu(),
        key.npu(),
        value.npu(),
        block_table.npu(),
        query_start,
        seq_len,
        prompt_len,
        scale,
    )
    call = (
        lambda: torch.ops.trianglemix_reference.
        triangle_paged_sparse_attention(*args)
    )

    for _ in range(warmup):
        call()
    torch.npu.synchronize()

    samples: list[float] = []
    for _ in range(repeats):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            call()
        end.record()
        torch.npu.synchronize()
        samples.append(float(start.elapsed_time(end)) / iterations)

    return {
        "name": name,
        "query_start": query_start,
        "query_tokens": query_tokens,
        "seq_len": seq_len,
        "prompt_len": prompt_len,
        "clock": "torch.npu.Event",
        "warmup": warmup,
        "iterations": iterations,
        "repeats": repeats,
        "samples_ms": samples,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
        "claim": "correctness-reference timing only",
    }


def main() -> None:
    args = parse_args()
    torch.ops.load_library(str(args.adapter.resolve()))
    cases = [
        ("dense_prefix", 0, 4, 1024, 1001),
        ("split_sink_local", 520, 4, 1024, 1002),
        ("dense_tail_boundary", 894, 4, 1024, 1003),
    ]
    result = {
        "warning": (
            "AIV scalar-reference latency; not a production performance claim"
        ),
        "cases": [
            time_case(
                *case,
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            for case in cases
        ],
    }
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
