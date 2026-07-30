#!/usr/bin/env python3
"""Paired dense-FIA versus TrianglePagedSparseAttention v2 crossover sweep.

The benchmark is intentionally self-contained and acquires the project-wide
NPU lock.  It measures the frozen custom-op call (no Python KV packing) against
the upstream dense paged TND FIA call on the same randomized BSND cache and
block table.  Measurement order alternates AB/BA both between cells and
between repeats to reduce thermal and process drift.

Run only after the production v2 custom operator and Torch adapter compile.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import statistics
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import torch
import torch_npu

from npu_v2_correctness import (
    _noncontiguous_block_table,
    run_gqa_sentinel,
    run_shared_softmax_sentinel,
)
from npu_v2_harness_common import (
    DEFAULT_NPU_LOCK,
    exclusive_npu_lock,
    file_sha256,
)
from triangle_v2_reference import (
    PRODUCTION_GEOMETRY,
    selected_intervals,
)
from v2_crossover_utils import summarize_crossover


GEOMETRY = PRODUCTION_GEOMETRY
SCALE = GEOMETRY.head_dim**-0.5
DEFAULT_SEQ_ENDS = [2048, 4096, 6144, 8192]
DEFAULT_QUERY_LENGTHS = [
    1,
    31,
    32,
    33,
    64,
    128,
    256,
    512,
    768,
    1024,
    1536,
    2048,
]


def event_sample_ms(
    function: Callable[[], object],
    iterations: int,
) -> float:
    start = torch.npu.Event(enable_timing=True)
    end = torch.npu.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / iterations


def memory_info() -> dict[str, int] | None:
    try:
        free_bytes, total_bytes = torch.npu.mem_get_info()
    except Exception:
        return None
    return {
        "free_bytes": int(free_bytes),
        "total_bytes": int(total_bytes),
        "used_bytes": int(total_bytes - free_bytes),
    }


def make_inputs(
    maximum_seq_end: int,
    maximum_query_len: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    logical_pages = math.ceil(maximum_seq_end / GEOMETRY.page_size)
    physical_pages = logical_pages + 17
    block_table = _noncontiguous_block_table(
        logical_pages,
        physical_pages,
        generator,
    )
    cache_shape = (
        physical_pages,
        GEOMETRY.page_size,
        GEOMETRY.kv_heads,
        GEOMETRY.head_dim,
    )
    key = (
        torch.randn(cache_shape, generator=generator)
        .mul_(0.5)
        .to(torch.bfloat16)
        .npu()
    )
    value = (
        torch.randn(cache_shape, generator=generator)
        .mul_(0.5)
        .to(torch.bfloat16)
        .npu()
    )
    query_bank = (
        torch.randn(
            maximum_query_len,
            GEOMETRY.query_heads,
            GEOMETRY.head_dim,
            generator=generator,
        )
        .mul_(0.5)
        .to(torch.bfloat16)
        .npu()
    )
    # FIA sparse_mode=3 uses bottom-right causal alignment for Tq != Tkv.
    # CANN 9.0.1's split-fuse path requires a 2048-column 2-D mask even when
    # every sampled query is shorter, so do not size this allocation solely
    # from the sweep's maximum query length.
    causal_mask_size = max(2048, maximum_query_len)
    causal_mask = torch.triu(
        torch.ones(
            causal_mask_size,
            causal_mask_size,
            dtype=torch.bool,
        ),
        diagonal=1,
    ).to(torch.int8).npu()
    return {
        "key": key,
        "value": value,
        "query_bank": query_bank,
        "block_table": block_table.npu(),
        "causal_mask": causal_mask,
    }


def dense_fia_call(
    query: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    seq_end: int,
) -> torch.Tensor:
    key = inputs["key"].view(
        inputs["key"].shape[0],
        GEOMETRY.page_size,
        -1,
    )
    value = inputs["value"].view(
        inputs["value"].shape[0],
        GEOMETRY.page_size,
        -1,
    )
    output, _ = torch_npu.npu_fused_infer_attention_score(
        query=query.contiguous(),
        key=key,
        value=value,
        atten_mask=inputs["causal_mask"],
        block_table=inputs["block_table"],
        input_layout="TND",
        block_size=GEOMETRY.page_size,
        actual_seq_lengths=[query.shape[0]],
        actual_seq_lengths_kv=[seq_end],
        num_key_value_heads=GEOMETRY.kv_heads,
        num_heads=GEOMETRY.query_heads,
        scale=SCALE,
        pre_tokens=2147483647,
        next_tokens=0,
        sparse_mode=3,
    )
    return output


def sparse_v2_call(
    query: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    seq_end: int,
    prompt_len: int,
) -> torch.Tensor:
    query_start = seq_end - query.shape[0]
    return torch.ops.trianglemix_reference.triangle_paged_sparse_attention(
        query.contiguous(),
        inputs["key"],
        inputs["value"],
        inputs["block_table"],
        query_start,
        seq_end,
        prompt_len,
        SCALE,
    )


def qk_work(
    seq_end: int,
    query_len: int,
    prompt_len: int,
) -> tuple[int, int]:
    query_start = seq_end - query_len
    dense_positions = 0
    sparse_positions = 0
    for row in range(query_len):
        query_position = query_start + row
        dense_positions += min(query_position + 1, seq_end)
        sparse_positions += sum(
            end - begin
            for begin, end in selected_intervals(
                query_position,
                seq_end,
                prompt_len,
                GEOMETRY,
            )
        )
    return dense_positions, sparse_positions


def benchmark_cell(
    *,
    seq_end: int,
    query_len: int,
    prompt_len: int,
    inputs: dict[str, torch.Tensor],
    warmup: int,
    iterations: int,
    repeats: int,
    cell_index: int,
) -> dict[str, Any]:
    query = inputs["query_bank"][:query_len]
    functions = {
        "dense": lambda: dense_fia_call(query, inputs, seq_end),
        "sparse": lambda: sparse_v2_call(
            query,
            inputs,
            seq_end,
            prompt_len,
        ),
    }
    for warmup_index in range(warmup):
        order = (
            ("dense", "sparse")
            if (cell_index + warmup_index) % 2 == 0
            else ("sparse", "dense")
        )
        for name in order:
            functions[name]()
    torch.npu.synchronize()

    samples: dict[str, list[float]] = {"dense": [], "sparse": []}
    repeat_orders: list[str] = []
    for repeat_index in range(repeats):
        order = (
            ("dense", "sparse")
            if (cell_index + repeat_index) % 2 == 0
            else ("sparse", "dense")
        )
        repeat_orders.append("->".join(order))
        for name in order:
            samples[name].append(
                event_sample_ms(functions[name], iterations)
            )

    dense_median = statistics.median(samples["dense"])
    sparse_median = statistics.median(samples["sparse"])
    robust_lower_bound = min(samples["dense"]) / max(samples["sparse"])
    dense_positions, sparse_positions = qk_work(
        seq_end,
        query_len,
        prompt_len,
    )
    return {
        "status": "ok",
        "seq_end": seq_end,
        "query_start": seq_end - query_len,
        "query_len": query_len,
        "prompt_len": prompt_len,
        "dense_qk_positions": dense_positions,
        "triangle_qk_positions": sparse_positions,
        "mathematical_qk_fraction": sparse_positions / dense_positions,
        "dense_ms": dense_median,
        "sparse_ms": sparse_median,
        "delta_ms": sparse_median - dense_median,
        "speedup": dense_median / sparse_median,
        "sparse_faster": sparse_median < dense_median,
        "robust_sparse_faster": robust_lower_bound > 1.0,
        "robust_speedup_lower_bound": robust_lower_bound,
        "dense_samples_ms": samples["dense"],
        "sparse_samples_ms": samples["sparse"],
        "repeat_orders": repeat_orders,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--seq-ends",
        type=int,
        nargs="+",
        default=DEFAULT_SEQ_ENDS,
    )
    parser.add_argument(
        "--query-lengths",
        type=int,
        nargs="+",
        default=DEFAULT_QUERY_LENGTHS,
    )
    parser.add_argument("--prompt-len", type=int, default=16384)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_NPU_LOCK)
    parser.add_argument("--lock-timeout", type=float, default=300.0)
    parser.add_argument(
        "--skip-semantic-smoke",
        action="store_true",
        help="Skip shared-softmax and GQA sentinels (not recommended)",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    seq_ends = sorted(set(args.seq_ends))
    query_lengths = sorted(set(args.query_lengths))
    if not seq_ends or seq_ends[0] <= 0:
        raise ValueError("seq ends must be positive")
    if not query_lengths or query_lengths[0] <= 0:
        raise ValueError("query lengths must be positive")
    if query_lengths[-1] > seq_ends[0]:
        raise ValueError("maximum query length exceeds the smallest seq end")
    if args.prompt_len < seq_ends[-1] + GEOMETRY.dense_tail:
        raise ValueError(
            "prompt_len must keep every measured row before the dense tail"
        )
    if args.warmup < 0 or args.iterations <= 0 or args.repeats <= 0:
        raise ValueError("invalid timing counts")
    return seq_ends, query_lengths


def main() -> int:
    args = parse_args()
    seq_ends, query_lengths = validate_args(args)
    adapter = args.adapter.resolve()
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    output = args.output.resolve()

    with exclusive_npu_lock(args.lock_path, args.lock_timeout):
        torch.npu.set_device(args.device)
        memory_before = memory_info()
        torch.ops.load_library(str(adapter))

        semantic_smoke: list[dict[str, Any]] = []
        if not args.skip_semantic_smoke:
            semantic_smoke = [
                run_shared_softmax_sentinel(),
                run_gqa_sentinel(),
            ]
            for result in semantic_smoke:
                print(json.dumps(result, ensure_ascii=False), flush=True)
            if any(result["status"] != "PASS" for result in semantic_smoke):
                raise AssertionError(
                    "semantic smoke failed; refusing to publish timings"
                )

        inputs = make_inputs(
            max(seq_ends),
            max(query_lengths),
            args.seed,
        )
        torch.npu.synchronize()
        memory_after_inputs = memory_info()

        # Put one-token probes last so an asynchronous rejection cannot
        # invalidate the main matrix.
        regular_lengths = [
            query_len for query_len in query_lengths if query_len != 1
        ]
        ordered_cells = [
            (seq_end, query_len)
            for seq_end in seq_ends
            for query_len in regular_lengths
        ]
        if 1 in query_lengths:
            ordered_cells.extend((seq_end, 1) for seq_end in seq_ends)

        records: list[dict[str, Any]] = []
        for cell_index, (seq_end, query_len) in enumerate(ordered_cells):
            try:
                record = benchmark_cell(
                    seq_end=seq_end,
                    query_len=query_len,
                    prompt_len=args.prompt_len,
                    inputs=inputs,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    cell_index=cell_index,
                )
            except Exception as error:
                record = {
                    "status": "error",
                    "seq_end": seq_end,
                    "query_start": seq_end - query_len,
                    "query_len": query_len,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)

        records.sort(
            key=lambda record: (
                int(record["seq_end"]),
                int(record["query_len"]),
            )
        )
        script_path = Path(__file__).resolve()
        report = {
            "schema_version": 1,
            "status": (
                "PASS"
                if all(record["status"] == "ok" for record in records)
                else "PARTIAL"
            ),
            "geometry": {
                **asdict(GEOMETRY),
                "prompt_len": args.prompt_len,
                "dense_baseline": "paged TND FIA sparse_mode=3",
                "sparse_candidate": (
                    "TrianglePagedSparseAttention v2 direct paged K/V, "
                    "no Python pack"
                ),
            },
            "measurement": {
                "clock": "torch.npu.Event",
                "warmup": args.warmup,
                "iterations_per_sample": args.iterations,
                "repeats": args.repeats,
                "order": "paired AB/BA alternating by cell and repeat",
                "robust_rule": (
                    "min(dense repeat) > max(sparse repeat)"
                ),
            },
            "versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(args.device),
            },
            "source": {
                "script": str(script_path),
                "script_sha256": file_sha256(script_path),
                "adapter": str(adapter),
                "adapter_sha256": file_sha256(adapter),
            },
            "lock_path": str(args.lock_path),
            "memory": {
                "before": memory_before,
                "after_inputs": memory_after_inputs,
            },
            "semantic_smoke": semantic_smoke,
            "seq_ends": seq_ends,
            "query_lengths": query_lengths,
            "records": records,
            "crossover": summarize_crossover(records, seq_ends),
            "created_unix_seconds": time.time(),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "CROSSOVER "
            + json.dumps(report["crossover"], ensure_ascii=False),
            flush=True,
        )
        print(f"result={output}", flush=True)
        return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
