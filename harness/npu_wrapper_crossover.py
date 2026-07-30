#!/usr/bin/env python3
"""ABBA latency sweep: direct wrapper versus one-shot dense paged FIA.

Run ``npu_wrapper_correctness.py`` successfully first.  This benchmark uses
the exact same candidate source, final ``.out`` adapter, randomized paged KV
cache, caller-owned outputs, and official dense FIA baseline.  NPU Event
samples alternate AB/BA, producing ABBA order across every adjacent repeat
pair.  Records expose query_start, Tq, prompt length, split kind, stage count,
and exact saved-QK work for crossover fitting.
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType
from typing import Any

import torch
import torch_npu

from harness_spec import (
    ALL_NAMED_SWEEP_CASES,
    DEFAULT_SWEEP_CASES,
    FALLBACK_FULL_SPARSE_FREEZE_CASES,
    FALLBACK_SPLIT_FREEZE_CASES,
    PROMPT_8320_CHUNK2048_ROUTE,
    WrapperCase,
    cross_product_cases,
    plan_for_case,
)
from npu_harness_common import (
    DEFAULT_NPU_LOCK,
    NpuInputs,
    dense_paged_fia_out,
    event_sample_ms,
    exclusive_npu_lock,
    file_sha256,
    load_candidate_module,
    make_npu_inputs,
    memory_info,
    metrics_within,
    run_wrapper_or_dense_fallback,
    sample_summary,
    standalone_single_launch_out,
    tensor_error_metrics,
)


DEFAULT_AXIS_QUERY_STARTS = [0, 500, 521, 1024, 2048, 4096]
DEFAULT_AXIS_QUERY_LENGTHS = [32, 128, 512, 2048]
DEFAULT_AXIS_PROMPT_LENGTHS = [1024, 4096, 8192]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wrapper", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        help="Select a named built-in sweep cell; may be repeated",
    )
    parser.add_argument(
        "--case-group",
        choices=(
            "default",
            "route-8320",
            "fallback-full",
            "fallback-split",
            "fallback-freeze",
        ),
        help="Select one curated case group instead of individual cells",
    )
    parser.add_argument(
        "--query-starts",
        type=int,
        nargs="+",
        help="Build an axis cross-product instead of built-in cells",
    )
    parser.add_argument(
        "--query-lengths",
        type=int,
        nargs="+",
        help="Tq axis used with --query-starts/--prompt-lengths",
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        help="Final-prompt axis used for the cross-product",
    )
    parser.add_argument(
        "--include-fallback",
        action="store_true",
        help="Also time cells with no sparse middle (dense vs dense)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="Use an even value so every repeat pair is exact ABBA",
    )
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--dense-region-max-abs",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--dense-region-mean-abs",
        type=float,
        default=0.004,
    )
    parser.add_argument(
        "--sparse-stage-max-abs",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=DEFAULT_NPU_LOCK,
    )
    parser.add_argument("--lock-timeout", type=float, default=300.0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def select_cases(args: argparse.Namespace) -> tuple[WrapperCase, ...]:
    axes_requested = any(
        value is not None
        for value in (
            args.query_starts,
            args.query_lengths,
            args.prompt_lengths,
        )
    )
    if axes_requested and (args.case or args.case_group):
        raise ValueError(
            "--case/--case-group cannot be combined with axis arguments"
        )
    if args.case and args.case_group:
        raise ValueError("--case cannot be combined with --case-group")
    if axes_requested:
        cases = cross_product_cases(
            args.query_starts or DEFAULT_AXIS_QUERY_STARTS,
            args.query_lengths or DEFAULT_AXIS_QUERY_LENGTHS,
            args.prompt_lengths or DEFAULT_AXIS_PROMPT_LENGTHS,
            seed=args.seed,
        )
    elif args.case:
        by_name = {case.name: case for case in ALL_NAMED_SWEEP_CASES}
        unknown = sorted(set(args.case) - set(by_name))
        if unknown:
            raise ValueError(f"unknown cases: {', '.join(unknown)}")
        cases = tuple(by_name[name] for name in args.case)
    elif args.case_group:
        groups = {
            "default": DEFAULT_SWEEP_CASES,
            "route-8320": PROMPT_8320_CHUNK2048_ROUTE,
            "fallback-full": (
                FALLBACK_FULL_SPARSE_FREEZE_CASES
                + PROMPT_8320_CHUNK2048_ROUTE[1:4]
            ),
            "fallback-split": (
                FALLBACK_SPLIT_FREEZE_CASES
                + (
                    PROMPT_8320_CHUNK2048_ROUTE[0],
                    PROMPT_8320_CHUNK2048_ROUTE[-1],
                )
            ),
            "fallback-freeze": (
                FALLBACK_FULL_SPARSE_FREEZE_CASES
                + FALLBACK_SPLIT_FREEZE_CASES
                + PROMPT_8320_CHUNK2048_ROUTE
            ),
        }
        cases = groups[args.case_group]
    else:
        cases = DEFAULT_SWEEP_CASES

    # The curated built-in route intentionally includes its final no-middle
    # cell: it is needed when summing all five attention calls for an
    # 8320-token prompt.  Axis sweeps still omit fallback cells unless the
    # caller explicitly requests them.
    if axes_requested and not args.include_fallback:
        cases = tuple(
            case
            for case in cases
            if plan_for_case(case).has_sparse_middle
        )
    if not cases:
        raise ValueError("no latency cells remain after filtering")
    return cases


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    if args.repeats <= 0 or args.repeats % 2:
        raise ValueError("repeats must be a positive even number for ABBA")
    if args.lock_timeout < 0:
        raise ValueError("lock timeout must be non-negative")
    for name in (
        "dense_region_max_abs",
        "dense_region_mean_abs",
        "sparse_stage_max_abs",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")


def lightweight_device_preflight(
    *,
    wrapper: ModuleType,
    config: object,
    case: WrapperCase,
    inputs: NpuInputs,
    dense_region_max_abs: float,
    dense_region_mean_abs: float,
    sparse_stage_max_abs: float,
) -> dict[str, Any]:
    """Gate wrapper composition without repeating the expensive CPU oracle."""
    plan = plan_for_case(case)
    query = inputs.query(case.query_tokens)
    dense_output = torch.empty_like(query)
    direct_output = torch.empty_like(query)
    lse_scratch = torch.empty(
        0,
        dtype=torch.float32,
        device=query.device,
    )
    dense_paged_fia_out(
        query=query,
        inputs=inputs,
        seq_len=case.seq_len,
        output=dense_output,
        softmax_lse_scratch=lse_scratch,
    )
    returned, mode = run_wrapper_or_dense_fallback(
        wrapper=wrapper,
        config=config,
        case=case,
        inputs=inputs,
        output=direct_output,
        softmax_lse_scratch=lse_scratch,
    )
    standalone_single_launch = standalone_single_launch_out(
        case=case,
        plan=plan,
        inputs=inputs,
    )
    torch.npu.synchronize()
    full_quality = tensor_error_metrics(direct_output, dense_output)
    regions: dict[str, dict[str, float | bool]] = {}
    checks = [
        returned is direct_output,
        mode == (
            "direct" if plan.has_sparse_middle else "dense_fallback"
        ),
        bool(full_quality["finite"]),
    ]
    if not plan.has_sparse_middle:
        checks.append(
            metrics_within(
                full_quality,
                max_abs=dense_region_max_abs,
                mean_abs=dense_region_mean_abs,
            )
        )
    else:
        if plan.q0 < plan.s0:
            prefix_rows = plan.s0 - plan.q0
            prefix = tensor_error_metrics(
                direct_output[:prefix_rows],
                dense_output[:prefix_rows],
            )
            regions["dense_prefix_vs_single_dense"] = prefix
            checks.append(
                metrics_within(
                    prefix,
                    max_abs=dense_region_max_abs,
                    mean_abs=dense_region_mean_abs,
                )
            )
        if plan.s1 < plan.q1:
            tail_start = plan.s1 - plan.q0
            tail = tensor_error_metrics(
                direct_output[tail_start:],
                dense_output[tail_start:],
            )
            regions["dense_tail_vs_single_dense"] = tail
            checks.append(
                metrics_within(
                    tail,
                    max_abs=dense_region_max_abs,
                    mean_abs=dense_region_mean_abs,
                )
            )
        assert standalone_single_launch is not None
        launch = tensor_error_metrics(
            direct_output,
            standalone_single_launch,
        )
        regions["full_vs_standalone_single_launch"] = launch
        checks.append(
            bool(launch["finite"])
            and float(launch["max_abs"]) <= sparse_stage_max_abs
        )
    return {
        "status": "PASS" if all(checks) else "FAIL",
        "mode": mode,
        "full_candidate_vs_single_dense": full_quality,
        "region_metrics": regions,
        "returned_original_output": returned is direct_output,
    }


def benchmark_cell(
    *,
    wrapper: ModuleType,
    config: object,
    case: WrapperCase,
    inputs: NpuInputs,
    warmup: int,
    iterations: int,
    repeats: int,
    cell_index: int,
) -> dict[str, Any]:
    plan = plan_for_case(case)
    query = inputs.query(case.query_tokens)
    dense_output = torch.empty_like(query)
    direct_output = torch.empty_like(query)
    lse_scratch = torch.empty(
        0,
        dtype=torch.float32,
        device=query.device,
    )

    def run_dense() -> torch.Tensor:
        return dense_paged_fia_out(
            query=query,
            inputs=inputs,
            seq_len=case.seq_len,
            output=dense_output,
            softmax_lse_scratch=lse_scratch,
        )

    def run_direct() -> torch.Tensor:
        returned, _ = run_wrapper_or_dense_fallback(
            wrapper=wrapper,
            config=config,
            case=case,
            inputs=inputs,
            output=direct_output,
            softmax_lse_scratch=lse_scratch,
        )
        return returned

    functions = {"dense": run_dense, "direct": run_direct}
    base_order = (
        ("dense", "direct")
        if cell_index % 2 == 0
        else ("direct", "dense")
    )
    for warmup_index in range(warmup):
        order = (
            base_order
            if warmup_index % 2 == 0
            else tuple(reversed(base_order))
        )
        for name in order:
            functions[name]()
    torch.npu.synchronize()

    samples: dict[str, list[float]] = {"dense": [], "direct": []}
    repeat_orders: list[str] = []
    for repeat_index in range(repeats):
        # Adjacent repeats are AB then BA: the launch stream is exact ABBA.
        order = (
            base_order
            if repeat_index % 2 == 0
            else tuple(reversed(base_order))
        )
        repeat_orders.append("->".join(order))
        for name in order:
            samples[name].append(
                event_sample_ms(functions[name], iterations)
            )

    dense_stats = sample_summary(samples["dense"])
    direct_stats = sample_summary(samples["direct"])
    dense_median = dense_stats["median_ms"]
    direct_median = direct_stats["median_ms"]
    robust_speedup = min(samples["dense"]) / max(samples["direct"])
    paired_dense_minus_direct = [
        dense_sample - direct_sample
        for dense_sample, direct_sample in zip(
            samples["dense"],
            samples["direct"],
        )
    ]
    paired_speedups = [
        dense_sample / direct_sample
        for dense_sample, direct_sample in zip(
            samples["dense"],
            samples["direct"],
        )
    ]
    conservative_delta = min(samples["dense"]) - max(samples["direct"])
    return {
        **case.to_dict(),
        "status": "ok",
        "plan": plan.to_dict(),
        "dense_ms": dense_median,
        "direct_ms": direct_median,
        "delta_ms": direct_median - dense_median,
        "speedup": dense_median / direct_median,
        "end_to_end_gain_percent": (
            (dense_median - direct_median) / dense_median * 100.0
        ),
        "direct_faster": direct_median < dense_median,
        "robust_speedup_lower_bound": robust_speedup,
        "robust_direct_faster": robust_speedup > 1.0,
        "conservative_dense_minus_direct_ms": conservative_delta,
        "conservative_gain_percent": (
            conservative_delta / min(samples["dense"]) * 100.0
        ),
        "dense_samples_ms": samples["dense"],
        "direct_samples_ms": samples["direct"],
        "dense_statistics": dense_stats,
        "direct_statistics": direct_stats,
        "paired_dense_minus_direct_samples_ms": paired_dense_minus_direct,
        "paired_dense_minus_direct_statistics": sample_summary(
            paired_dense_minus_direct
        ),
        "paired_speedup_samples": paired_speedups,
        "paired_speedup_median": statistics.median(paired_speedups),
        "paired_speedup_min": min(paired_speedups),
        "paired_speedup_max": max(paired_speedups),
        "any_tukey_outlier": bool(
            dense_stats["tukey_outlier_count"]
            or direct_stats["tukey_outlier_count"]
        ),
        "repeat_orders": repeat_orders,
    }


def summarize_crossover(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    valid = [record for record in records if record["status"] == "ok"]
    by_split: dict[str, dict[str, Any]] = {}
    for split_kind in sorted(
        {
            str(record["plan"]["split_kind"])
            for record in valid
        }
    ):
        group = [
            record
            for record in valid
            if record["plan"]["split_kind"] == split_kind
        ]
        faster = [record for record in group if record["direct_faster"]]
        robust = [
            record
            for record in group
            if record["robust_direct_faster"]
        ]
        by_split[split_kind] = {
            "cells": len(group),
            "direct_faster_cells": len(faster),
            "robust_direct_faster_cells": len(robust),
            "median_speedup": statistics.median(
                float(record["speedup"]) for record in group
            ),
            "minimum_saved_qk_direct_faster": (
                min(int(record["plan"]["saved_qk"]) for record in faster)
                if faster
                else None
            ),
            "minimum_saved_qk_robust_direct_faster": (
                min(int(record["plan"]["saved_qk"]) for record in robust)
                if robust
                else None
            ),
        }
    robust_all = [
        record for record in valid if record["robust_direct_faster"]
    ]
    return {
        "valid_cells": len(valid),
        "direct_faster_cells": sum(
            bool(record["direct_faster"]) for record in valid
        ),
        "robust_direct_faster_cells": len(robust_all),
        "minimum_saved_qk_robust_direct_faster": (
            min(
                int(record["plan"]["saved_qk"])
                for record in robust_all
            )
            if robust_all
            else None
        ),
        "by_split_kind": by_split,
    }


def summarize_prompt_8320_route(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Sum the five measured chunk calls used by the 8320-token TTFT route."""
    expected_names = tuple(
        case.name for case in PROMPT_8320_CHUNK2048_ROUTE
    )
    by_name = {str(record["name"]): record for record in records}
    present = [
        by_name[name] for name in expected_names if name in by_name
    ]
    valid = [
        record for record in present if record.get("status") == "ok"
    ]
    chunks: list[dict[str, Any]] = []
    for record in present:
        if record.get("status") != "ok":
            chunks.append(
                {
                    "name": record["name"],
                    "status": record.get("status", "missing"),
                }
            )
            continue
        plan = record["plan"]
        direct_faster = bool(record["direct_faster"])
        robust_direct_faster = bool(record["robust_direct_faster"])
        chunks.append(
            {
                "name": record["name"],
                "status": "ok",
                "query_start": record["query_start"],
                "query_tokens": record["query_tokens"],
                "split_kind": plan["split_kind"],
                "stage_count": plan["stage_count"],
                "saved_qk": plan["saved_qk"],
                "dense_ms": record["dense_ms"],
                "direct_ms": record["direct_ms"],
                "median_dispatch": (
                    "direct" if direct_faster else "dense_fallback"
                ),
                "robust_dispatch": (
                    "direct"
                    if robust_direct_faster
                    else "dense_fallback"
                ),
            }
        )

    complete = (
        len(present) == len(expected_names)
        and len(valid) == len(expected_names)
    )
    summary: dict[str, Any] = {
        "prompt_len": 8320,
        "nominal_chunk_size": 2048,
        "expected_cells": list(expected_names),
        "complete": complete,
        "chunks": chunks,
    }
    if not complete:
        summary["missing_or_failed_cells"] = [
            name
            for name in expected_names
            if name not in by_name or by_name[name].get("status") != "ok"
        ]
        return summary

    dense_sum = sum(float(record["dense_ms"]) for record in valid)
    direct_sum = sum(float(record["direct_ms"]) for record in valid)
    median_policy_sum = sum(
        min(float(record["dense_ms"]), float(record["direct_ms"]))
        for record in valid
    )
    robust_policy_sum = sum(
        (
            float(record["direct_ms"])
            if record["robust_direct_faster"]
            else float(record["dense_ms"])
        )
        for record in valid
    )
    summary.update(
        {
            "single_layer_dense_attention_sum_ms": dense_sum,
            "single_layer_always_direct_wrapper_sum_ms": direct_sum,
            "single_layer_median_policy_sum_ms": median_policy_sum,
            "single_layer_robust_policy_sum_ms": robust_policy_sum,
            "always_direct_gain_percent": (
                (dense_sum - direct_sum) / dense_sum * 100.0
            ),
            "median_policy_gain_percent": (
                (dense_sum - median_policy_sum) / dense_sum * 100.0
            ),
            "robust_policy_gain_percent": (
                (dense_sum - robust_policy_sum) / dense_sum * 100.0
            ),
            "structural_fallback_chunks": [
                record["name"]
                for record in valid
                if record["plan"]["split_kind"] == "dense_fallback"
            ],
            "median_policy_fallback_chunks": [
                record["name"]
                for record in valid
                if not record["direct_faster"]
            ],
            "robust_policy_fallback_chunks": [
                record["name"]
                for record in valid
                if not record["robust_direct_faster"]
            ],
        }
    )
    return summary


def main() -> int:
    args = parse_args()
    validate_args(args)
    cases = select_cases(args)
    wrapper_path = args.wrapper.resolve()
    adapter_path = args.adapter.resolve()
    output_path = args.output.resolve()
    if not wrapper_path.is_file():
        raise FileNotFoundError(wrapper_path)
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    records: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = []
    started = time.time()
    memory_before = None
    memory_after_inputs = None
    with exclusive_npu_lock(args.lock_path, args.lock_timeout):
        torch.npu.set_device(args.device)
        memory_before = memory_info()
        wrapper, config = load_candidate_module(
            wrapper_path,
            adapter_path,
        )
        inputs = make_npu_inputs(
            maximum_seq_len=max(case.seq_len for case in cases),
            maximum_query_tokens=max(
                case.query_tokens for case in cases
            ),
            seed=args.seed,
        )
        torch.npu.synchronize()
        memory_after_inputs = memory_info()

        for cell_index, case in enumerate(cases):
            try:
                preflight = lightweight_device_preflight(
                    wrapper=wrapper,
                    config=config,
                    case=case,
                    inputs=inputs,
                    dense_region_max_abs=args.dense_region_max_abs,
                    dense_region_mean_abs=args.dense_region_mean_abs,
                    sparse_stage_max_abs=args.sparse_stage_max_abs,
                )
                preflight_record = {
                    "name": case.name,
                    **preflight,
                }
                preflights.append(preflight_record)
                if preflight["status"] != "PASS":
                    raise AssertionError(
                        f"{case.name}: device composition preflight failed"
                    )
                record = benchmark_cell(
                    wrapper=wrapper,
                    config=config,
                    case=case,
                    inputs=inputs,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    cell_index=cell_index,
                )
            except Exception as error:
                record = {
                    **case.to_dict(),
                    "status": "error",
                    "plan": plan_for_case(case).to_dict(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if (
                record["status"] != "ok"
                and not args.continue_on_error
            ):
                break
        torch.npu.synchronize()

    all_ok = (
        len(records) == len(cases)
        and all(record["status"] == "ok" for record in records)
    )
    summary = summarize_crossover(records)
    route_summary = summarize_prompt_8320_route(records)
    script_path = Path(__file__).resolve()
    common_path = script_path.with_name("npu_harness_common.py")
    spec_path = script_path.with_name("harness_spec.py")
    report = {
        "schema_version": 1,
        "suite": "trianglemix_direct_wrapper_crossover",
        "status": "PASS" if all_ok else "PARTIAL",
        "correctness_prerequisite": (
            "npu_wrapper_correctness.py must pass first; per-cell preflight "
            "rechecks dense slices and standalone custom middle"
        ),
        "geometry": {
            "dtype": "bfloat16",
            "query_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "page_size": 128,
            "batch": 1,
            "randomized_physical_pages": True,
        },
        "measurement": {
            "clock": "torch.npu.Event",
            "warmup": args.warmup,
            "iterations_per_sample": args.iterations,
            "repeats": args.repeats,
            "order": (
                "paired AB/BA; every adjacent repeat pair is ABBA; "
                "starting side alternates by cell"
            ),
            "synchronization": (
                "pre-warmup torch.npu.synchronize plus Event end synchronize"
            ),
            "baseline": "one-shot official dense paged FIA .out",
            "candidate": "complete direct wrapper including split launches",
        },
        "versions": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "torch_npu": torch_npu.__version__,
            "device": torch.npu.get_device_name(args.device),
        },
        "source": {
            "wrapper": str(wrapper_path),
            "wrapper_sha256": file_sha256(wrapper_path),
            "adapter": str(adapter_path),
            "adapter_sha256": file_sha256(adapter_path),
            "script": str(script_path),
            "script_sha256": file_sha256(script_path),
            "common_sha256": file_sha256(common_path),
            "spec_sha256": file_sha256(spec_path),
        },
        "lock_path": str(args.lock_path),
        "memory": {
            "before": memory_before,
            "after_inputs": memory_after_inputs,
        },
        "cells": [case.to_dict() for case in cases],
        "preflight": preflights,
        "records": records,
        "crossover": summary,
        "prompt_8320_chunk2048_route": route_summary,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        "CROSSOVER "
        + json.dumps(summary, ensure_ascii=False),
        flush=True,
    )
    print(
        "ROUTE_8320 "
        + json.dumps(route_summary, ensure_ascii=False),
        flush=True,
    )
    print(f"result={output_path}", flush=True)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
