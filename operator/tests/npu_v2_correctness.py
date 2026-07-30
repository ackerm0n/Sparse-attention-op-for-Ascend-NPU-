#!/usr/bin/env python3
"""Independent NPU correctness matrix for the production v2 candidate.

Run this only after the v2 custom operator and Torch adapter have been built.
The script acquires the shared TriangleMix NPU lock, loads the frozen
batch-one adapter schema, and compares every output against the CPU oracle in
``triangle_v2_reference.py``.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch_npu  # noqa: F401

from npu_v2_harness_common import (
    DEFAULT_NPU_LOCK,
    exclusive_npu_lock,
    file_sha256,
)
from triangle_v2_reference import (
    PRODUCTION_GEOMETRY,
    physical_location,
    torch_reference_attention,
)


GEOMETRY = PRODUCTION_GEOMETRY
SCALE = GEOMETRY.head_dim**-0.5


@dataclass(frozen=True)
class CorrectnessCase:
    name: str
    query_start: int
    query_tokens: int
    seq_len: int
    prompt_len: int
    seed: int
    coverage: str


DEFAULT_CASES = (
    CorrectnessCase(
        "outer_tail_255",
        250,
        5,
        255,
        2048,
        2095,
        "255-column boundary mask within one 512-token outer tile",
    ),
    CorrectnessCase(
        "exact_outer_256",
        251,
        5,
        256,
        2048,
        2096,
        "exact 256-column inner boundary within one 512-token outer tile",
    ),
    CorrectnessCase(
        "second_outer_257",
        252,
        5,
        257,
        2048,
        2097,
        "256+1 columns within one outer tile",
    ),
    CorrectnessCase(
        "outer_tail_511",
        506,
        5,
        511,
        2048,
        2095,
        "511-column boundary mask in one outer tile",
    ),
    CorrectnessCase(
        "exact_outer_512",
        507,
        5,
        512,
        2048,
        2096,
        "exact outer tile and dedicated 512-column reduction path",
    ),
    CorrectnessCase(
        "second_outer_513",
        508,
        5,
        513,
        2048,
        2097,
        "512+1 outer tiles with shared online-softmax state",
    ),
    CorrectnessCase(
        "two_outer_tail_1023",
        1018,
        5,
        1023,
        2048,
        2098,
        "512+511 outer tiles with a boundary-masked tail",
    ),
    CorrectnessCase(
        "exact_two_outer_1024",
        1019,
        5,
        1024,
        2048,
        2099,
        "two exact 512-column outer tiles",
    ),
    CorrectnessCase(
        "third_outer_1025",
        1020,
        5,
        1025,
        2048,
        2100,
        "512+512+1 outer tiles with online accumulation",
    ),
    CorrectnessCase(
        "cross_page_127_128",
        126,
        5,
        131,
        2048,
        2101,
        "dense prefix, logical page boundary",
    ),
    CorrectnessCase(
        "sparse_begin_520_521",
        519,
        4,
        523,
        2048,
        2102,
        "sink/local overlap then first separated row",
    ),
    CorrectnessCase(
        "q_tile_boundary_with_sparse_begin",
        500,
        65,
        565,
        2048,
        2103,
        "two q_tile boundaries, ragged final row, sparseBegin",
    ),
    CorrectnessCase(
        "split_local_crosses_pages",
        766,
        35,
        801,
        2048,
        2104,
        "separated sink/local, local range crosses pages, ragged q tile",
    ),
    CorrectnessCase(
        "dense_tail_boundary_895_896",
        894,
        4,
        898,
        1024,
        2105,
        "last sparse rows followed by first dense-tail rows",
    ),
    CorrectnessCase(
        "last_dense_tail_row",
        1023,
        1,
        1024,
        1024,
        2106,
        "last of exactly 128 dense tail rows",
    ),
    CorrectnessCase(
        "ragged_query_31",
        700,
        31,
        731,
        2048,
        2107,
        "ragged query shorter than q_tile",
    ),
    CorrectnessCase(
        "ragged_query_33",
        700,
        33,
        733,
        2048,
        2108,
        "one full q_tile plus one row",
    ),
    CorrectnessCase(
        "long_random_paged_8192",
        8127,
        65,
        8192,
        16384,
        2109,
        "long-sequence sparse path and randomized non-contiguous pages",
    ),
)


def _noncontiguous_block_table(
    logical_pages: int,
    physical_pages: int,
    generator: torch.Generator,
) -> torch.Tensor:
    permutation = torch.randperm(
        physical_pages,
        generator=generator,
        dtype=torch.int64,
    )
    table = permutation[:logical_pages].clone()
    if logical_pages > 1 and bool(torch.all(table[1:] == table[:-1] + 1)):
        table[0], table[-1] = table[-1].clone(), table[0].clone()
    return table.to(torch.int32).view(1, -1).contiguous()


def make_random_inputs(
    case: CorrectnessCase,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if case.query_start + case.query_tokens > case.seq_len:
        raise ValueError(f"{case.name}: query extends beyond seq_len")
    generator = torch.Generator().manual_seed(case.seed)
    logical_pages = math.ceil(case.seq_len / GEOMETRY.page_size)
    physical_pages = logical_pages + 7
    block_table = _noncontiguous_block_table(
        logical_pages,
        physical_pages,
        generator,
    )
    query = (
        torch.randn(
            case.query_tokens,
            GEOMETRY.query_heads,
            GEOMETRY.head_dim,
            generator=generator,
        )
        .mul_(0.5)
        .to(torch.bfloat16)
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
    )
    value = (
        torch.randn(cache_shape, generator=generator)
        .mul_(0.5)
        .to(torch.bfloat16)
    )
    return query, key, value, block_table


def custom_call(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    block_table: torch.Tensor,
    query_start: int,
    seq_len: int,
    prompt_len: int,
) -> torch.Tensor:
    return torch.ops.trianglemix_reference.triangle_paged_sparse_attention(
        query.npu(),
        key.npu(),
        value.npu(),
        block_table.npu(),
        query_start,
        seq_len,
        prompt_len,
        SCALE,
    )


def error_metrics(
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> dict[str, float | bool]:
    actual = actual.float()
    expected = expected.float()
    difference = (actual - expected).abs()
    expected_norm = float(torch.linalg.vector_norm(expected))
    difference_norm = float(torch.linalg.vector_norm(difference))
    return {
        "finite": bool(
            torch.isfinite(actual).all() and torch.isfinite(expected).all()
        ),
        "max_abs": float(difference.max()),
        "mean_abs": float(difference.mean()),
        "relative_l2": difference_norm / max(expected_norm, 1.0e-12),
    }


def metrics_pass(
    metrics: dict[str, float | bool],
    *,
    max_abs_tolerance: float,
    mean_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> bool:
    return bool(
        metrics["finite"]
        and float(metrics["max_abs"]) <= max_abs_tolerance
        and float(metrics["mean_abs"]) <= mean_abs_tolerance
        and float(metrics["relative_l2"]) <= relative_l2_tolerance
    )


def run_random_case(
    case: CorrectnessCase,
    *,
    max_abs_tolerance: float,
    mean_abs_tolerance: float,
    relative_l2_tolerance: float,
) -> dict[str, Any]:
    query, key, value, block_table = make_random_inputs(case)
    expected = torch_reference_attention(
        query,
        key,
        value,
        block_table,
        case.query_start,
        case.seq_len,
        case.prompt_len,
        SCALE,
    )
    started = time.perf_counter()
    actual = custom_call(
        query,
        key,
        value,
        block_table,
        case.query_start,
        case.seq_len,
        case.prompt_len,
    )
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    metrics = error_metrics(actual.cpu(), expected)
    passed = metrics_pass(
        metrics,
        max_abs_tolerance=max_abs_tolerance,
        mean_abs_tolerance=mean_abs_tolerance,
        relative_l2_tolerance=relative_l2_tolerance,
    )
    logical_pages = math.ceil(case.seq_len / GEOMETRY.page_size)
    table_list = block_table[0, :logical_pages].tolist()
    return {
        **asdict(case),
        "status": "PASS" if passed else "FAIL",
        "metrics": metrics,
        "wall_ms_including_first_call": elapsed_ms,
        "logical_pages": logical_pages,
        "physical_table": table_list,
        "table_has_nonconsecutive_transition": any(
            right != left + 1
            for left, right in zip(table_list, table_list[1:])
        ),
    }


def _set_logical_values(
    cache: torch.Tensor,
    block_table: torch.Tensor,
    tokens: range | tuple[int, ...],
    value: float,
) -> None:
    table = block_table[0].tolist()
    for token in tokens:
        physical_page, offset = physical_location(
            token,
            table,
            GEOMETRY,
        )
        cache[physical_page, offset].fill_(value)


def run_shared_softmax_sentinel() -> dict[str, Any]:
    """Distinguish one joint softmax (~1.138) from two added softmaxes (11)."""

    query_start = 521
    seq_len = 522
    prompt_len = 2048
    logical_pages = math.ceil(seq_len / GEOMETRY.page_size)
    physical_pages = logical_pages + 5
    generator = torch.Generator().manual_seed(2201)
    block_table = _noncontiguous_block_table(
        logical_pages,
        physical_pages,
        generator,
    )
    query = torch.zeros(
        1,
        GEOMETRY.query_heads,
        GEOMETRY.head_dim,
        dtype=torch.bfloat16,
    )
    cache_shape = (
        physical_pages,
        GEOMETRY.page_size,
        GEOMETRY.kv_heads,
        GEOMETRY.head_dim,
    )
    key = torch.zeros(cache_shape, dtype=torch.bfloat16)
    value = torch.zeros(cache_shape, dtype=torch.bfloat16)
    _set_logical_values(value, block_table, tuple(range(8)), 10.0)
    # Poison the first excluded token.  It catches both an interval-boundary
    # error and a fake dense masked implementation that leaks the gap.
    _set_logical_values(value, block_table, (8,), 1000.0)
    _set_logical_values(value, block_table, range(9, 522), 1.0)

    expected = torch_reference_attention(
        query,
        key,
        value,
        block_table,
        query_start,
        seq_len,
        prompt_len,
        SCALE,
    )
    actual = custom_call(
        query,
        key,
        value,
        block_table,
        query_start,
        seq_len,
        prompt_len,
    )
    torch.npu.synchronize()
    actual_cpu = actual.cpu().float()
    metrics = error_metrics(actual_cpu, expected)
    joint_value = (8.0 * 10.0 + 513.0) / 521.0
    forbidden_two_softmax_value = 11.0
    representative = float(actual_cpu[0, 0, 0])
    passed = (
        bool(metrics["finite"])
        and float(metrics["max_abs"]) <= 0.04
        and abs(representative - joint_value) <= 0.04
        and abs(representative - forbidden_two_softmax_value) >= 1.0
    )
    return {
        "name": "shared_softmax_sentinel",
        "status": "PASS" if passed else "FAIL",
        "actual_representative": representative,
        "joint_softmax_expected": joint_value,
        "forbidden_two_softmax_sum": forbidden_two_softmax_value,
        "metrics": metrics,
    }


def run_gqa_sentinel() -> dict[str, Any]:
    """Verify exact Q-head to KV-head mapping for GQA 4:1."""

    generator = torch.Generator().manual_seed(2202)
    block_table = _noncontiguous_block_table(1, 4, generator)
    query = torch.zeros(
        1,
        GEOMETRY.query_heads,
        GEOMETRY.head_dim,
        dtype=torch.bfloat16,
    )
    cache_shape = (
        4,
        GEOMETRY.page_size,
        GEOMETRY.kv_heads,
        GEOMETRY.head_dim,
    )
    key = torch.zeros(cache_shape, dtype=torch.bfloat16)
    value = torch.zeros(cache_shape, dtype=torch.bfloat16)
    physical_page = int(block_table[0, 0])
    for kv_head in range(GEOMETRY.kv_heads):
        value[physical_page, 0, kv_head].fill_(float(kv_head + 1))
    actual = custom_call(
        query,
        key,
        value,
        block_table,
        query_start=0,
        seq_len=1,
        prompt_len=1024,
    )
    torch.npu.synchronize()
    representative = actual.cpu().float()[0, :, 0]
    expected = torch.tensor(
        [
            float(kv_head + 1)
            for kv_head in range(GEOMETRY.kv_heads)
            for _ in range(GEOMETRY.group_size)
        ]
    )
    maximum_error = float((representative - expected).abs().max())
    return {
        "name": "gqa_4_to_1_sentinel",
        "status": "PASS" if maximum_error <= 0.01 else "FAIL",
        "actual_per_query_head": representative.tolist(),
        "expected_per_query_head": expected.tolist(),
        "max_abs": maximum_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Path to triangle_paged_attention_torch*.so built against v2",
    )
    parser.add_argument(
        "--case",
        action="append",
        help="Run only a named randomized case; may be repeated",
    )
    parser.add_argument(
        "--only-sentinel",
        action="append",
        choices=("shared_softmax", "gqa"),
        help=(
            "Run only the selected sentinel; may be repeated. "
            "No randomized cases are run when this option is present."
        ),
    )
    parser.add_argument(
        "--skip-sentinels",
        action="store_true",
        help="Run selected randomized cases without either sentinel.",
    )
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_NPU_LOCK)
    parser.add_argument("--lock-timeout", type=float, default=300.0)
    parser.add_argument("--max-abs", type=float, default=0.05)
    parser.add_argument("--mean-abs", type=float, default=0.005)
    parser.add_argument("--relative-l2", type=float, default=0.02)
    parser.add_argument("--cpu-threads", type=int, default=8)
    return parser.parse_args()


def select_cases(names: list[str] | None) -> tuple[CorrectnessCase, ...]:
    if not names:
        return DEFAULT_CASES
    by_name = {case.name: case for case in DEFAULT_CASES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def main() -> int:
    args = parse_args()
    adapter = args.adapter.resolve()
    if not adapter.is_file():
        raise FileNotFoundError(adapter)
    if args.cpu_threads <= 0:
        raise ValueError("--cpu-threads must be positive")
    if args.only_sentinel and args.skip_sentinels:
        raise ValueError(
            "--only-sentinel and --skip-sentinels are mutually exclusive"
        )
    cases = () if args.only_sentinel else select_cases(args.case)
    sentinel_by_name = {
        "shared_softmax": run_shared_softmax_sentinel,
        "gqa": run_gqa_sentinel,
    }
    if args.skip_sentinels:
        sentinels = ()
    elif args.only_sentinel:
        sentinels = tuple(
            sentinel_by_name[name] for name in args.only_sentinel
        )
    else:
        sentinels = tuple(sentinel_by_name.values())
    torch.set_num_threads(args.cpu_threads)

    with exclusive_npu_lock(args.lock_path, args.lock_timeout):
        torch.npu.set_device(args.device)
        torch.ops.load_library(str(adapter))
        results: list[dict[str, Any]] = []
        for sentinel in sentinels:
            try:
                result = sentinel()
            except Exception as error:
                result = {
                    "name": sentinel.__name__,
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

        for case in cases:
            try:
                result = run_random_case(
                    case,
                    max_abs_tolerance=args.max_abs,
                    mean_abs_tolerance=args.mean_abs,
                    relative_l2_tolerance=args.relative_l2,
                )
            except Exception as error:
                result = {
                    **asdict(case),
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            results.append(result)
            print(json.dumps(result, ensure_ascii=False), flush=True)

        report = {
            "schema_version": 1,
            "status": (
                "PASS"
                if all(result["status"] == "PASS" for result in results)
                else "FAIL"
            ),
            "geometry": asdict(GEOMETRY),
            "tolerances": {
                "max_abs": args.max_abs,
                "mean_abs": args.mean_abs,
                "relative_l2": args.relative_l2,
            },
            "versions": {
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "torch_npu": torch_npu.__version__,
                "device": torch.npu.get_device_name(args.device),
            },
            "source": {
                "script": str(Path(__file__).resolve()),
                "script_sha256": file_sha256(Path(__file__).resolve()),
                "adapter": str(adapter),
                "adapter_sha256": file_sha256(adapter),
            },
            "lock_path": str(args.lock_path),
            "results": results,
            "pass_count": sum(
                result["status"] == "PASS" for result in results
            ),
            "fail_count": sum(
                result["status"] != "PASS" for result in results
            ),
        }
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        print(
            "NPU_V2_CORRECTNESS_"
            + report["status"]
            + f" pass={report['pass_count']} fail={report['fail_count']}",
            flush=True,
        )
        return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
