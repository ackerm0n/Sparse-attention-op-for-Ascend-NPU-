#!/usr/bin/env python3
"""B=1 NPU Event ABBA crossover over prompt-length × chunk-size routes."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from .common import (
    environment_fingerprint,
    sha256_file,
    write_json,
)
from .installed_wheel_correctness import _installed_provenance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = PROJECT_ROOT / "harness"


def _integer_list(value: str, *, name: str) -> list[int]:
    try:
        result = [
            int(item.strip())
            for item in value.split(",")
            if item.strip()
        ]
    except ValueError as error:
        raise ValueError(f"{name} must be comma-separated integers") from error
    if not result or any(item <= 0 for item in result):
        raise ValueError(f"{name} values must be positive")
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique")
    return result


def build_route_cases(
    *,
    lengths: list[int],
    chunk_sizes: list[int],
    seed: int,
    wrapper_case_type: type[Any],
) -> tuple[Any, ...]:
    cases: list[Any] = []
    index = 0
    for prompt_len in lengths:
        for chunk_size in chunk_sizes:
            for query_start in range(0, prompt_len, chunk_size):
                query_tokens = min(chunk_size, prompt_len - query_start)
                cases.append(
                    wrapper_case_type(
                        name=(
                            f"route_p{prompt_len}_c{chunk_size}_"
                            f"q0_{query_start}_tq_{query_tokens}"
                        ),
                        query_start=query_start,
                        query_tokens=query_tokens,
                        prompt_len=prompt_len,
                        seed=seed + index,
                        coverage=(
                            "length-by-chunk scheduler route crossover cell"
                        ),
                    )
                )
                index += 1
    return tuple(cases)


def _is_exact_abba(orders: object) -> bool:
    if not isinstance(orders, list) or not orders or len(orders) % 2:
        return False
    for index in range(0, len(orders), 2):
        first = str(orders[index]).split("->")
        second = str(orders[index + 1]).split("->")
        if len(first) != 2 or second != list(reversed(first)):
            return False
    return True


def _normalize_attention_record(
    record: dict[str, Any],
) -> dict[str, Any]:
    """Rename the legacy harness metric without leaking a TTFT claim."""

    legacy_name = "end_to_end_gain_percent"
    if legacy_name not in record:
        raise ValueError(
            "benchmark record is missing its legacy gain measurement"
        )
    if "attention_gain_percent" in record:
        raise ValueError("benchmark record contains both gain field names")
    value = record.pop(legacy_name)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
    ):
        raise TypeError("benchmark gain measurement must be numeric")
    record["attention_gain_percent"] = float(value)
    return record


def _load_correctness_prerequisite(
    path: Path,
    wheel_sha256: str,
) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("correctness report root must be an object")
    if (
        value.get("suite")
        != "trianglemix_installed_wheel_npu_correctness"
        or value.get("status") != "PASS"
        or not isinstance(value.get("wheel"), dict)
        or value["wheel"].get("sha256") != wheel_sha256
    ):
        raise ValueError(
            "correctness prerequisite must be PASS for the exact --wheel"
        )
    return {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "suite": value.get("suite"),
        "status": value.get("status"),
        "wheel_sha256": value["wheel"].get("sha256"),
    }


def _production_route(
    kernel: object,
    production_config: object,
    case: object,
) -> bool:
    return bool(
        kernel.triangle_direct_eligible(
            query_len=case.query_tokens,
            seq_len=case.seq_len,
            prompt_len=case.prompt_len,
            config=production_config,
        )
    )


def summarize_routes(
    records: list[dict[str, Any]],
    *,
    lengths: list[int],
    chunk_sizes: list[int],
) -> list[dict[str, Any]]:
    routes: list[dict[str, Any]] = []
    for prompt_len in lengths:
        for chunk_size in chunk_sizes:
            cells = [
                record
                for record in records
                if record.get("prompt_len") == prompt_len
                and record.get("chunk_size") == chunk_size
            ]
            complete = (
                bool(cells)
                and sum(int(item["query_tokens"]) for item in cells)
                == prompt_len
                and all(item.get("status") == "ok" for item in cells)
            )
            route: dict[str, Any] = {
                "prompt_len": prompt_len,
                "chunk_size": chunk_size,
                "batch_size": 1,
                "complete": complete,
                "cell_count": len(cells),
                "cell_names": [item.get("name") for item in cells],
            }
            if complete:
                dense = sum(float(item["dense_ms"]) for item in cells)
                policy = sum(
                    (
                        float(item["direct_ms"])
                        if item["release_routed_direct"]
                        else float(item["dense_ms"])
                    )
                    for item in cells
                )
                route.update(
                    {
                        "dense_attention_sum_ms": dense,
                        "release_policy_attention_sum_ms": policy,
                        "release_policy_attention_gain_percent": (
                            (dense - policy) / dense * 100.0
                        ),
                        "release_direct_cells": sum(
                            bool(item["release_routed_direct"])
                            for item in cells
                        ),
                        "release_false_positive_direct_cells": sum(
                            bool(item["release_false_positive_direct"])
                            for item in cells
                        ),
                    }
                )
            routes.append(route)
    return routes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument(
        "--correctness-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--lengths",
        default="8192,8193,8320",
    )
    parser.add_argument(
        "--chunk-sizes",
        default="512,2048",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--repeats",
        type=int,
        default=8,
        help="Positive even count; each adjacent pair is exact ABBA.",
    )
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--lock-timeout", type=float, default=300.0)
    parser.add_argument("--dense-region-max-abs", type=float, default=0.04)
    parser.add_argument("--dense-region-mean-abs", type=float, default=0.004)
    parser.add_argument("--sparse-stage-max-abs", type=float, default=0.0)
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_path = args.output.resolve()
    wheel_path = args.wheel.resolve()
    correctness_path = args.correctness_report.resolve()
    lengths = _integer_list(args.lengths, name="--lengths")
    chunk_sizes = _integer_list(
        args.chunk_sizes,
        name="--chunk-sizes",
    )
    if max(chunk_sizes) > 2048:
        raise SystemExit(
            "the validated harness contract supports chunk sizes <= 2048"
        )
    if args.warmup < 0 or args.iterations <= 0:
        raise SystemExit("warmup must be non-negative and iterations positive")
    if args.repeats <= 0 or args.repeats % 2:
        raise SystemExit("--repeats must be positive and even")
    if args.lock_timeout < 0:
        raise SystemExit("--lock-timeout must be non-negative")
    if not wheel_path.is_file() or not correctness_path.is_file():
        raise SystemExit("wheel and correctness report must exist")

    wheel_sha256 = sha256_file(wheel_path)
    prerequisite = _load_correctness_prerequisite(
        correctness_path,
        wheel_sha256,
    )
    sys.path.insert(0, str(HARNESS_DIR))
    import torch
    import torch_npu
    from harness_spec import WrapperCase, plan_for_case
    from npu_harness_common import (
        DEFAULT_NPU_LOCK,
        exclusive_npu_lock,
        make_npu_inputs,
        memory_info,
    )
    from npu_wrapper_crossover import (
        benchmark_cell,
        lightweight_device_preflight,
    )

    kernel = importlib.import_module("vllm_ascend_trianglemix.kernel")
    native = importlib.import_module("vllm_ascend_trianglemix.native")
    provenance = _installed_provenance(kernel, native, wheel_path)
    if (
        not provenance["is_installed_non_source_copy"]
        or not provenance["exact_wheel_payload_match"]["passed"]
    ):
        raise RuntimeError(
            "TriangleMix imports do not exactly match the supplied wheel"
        )
    native_status_object = native.ensure_native_loaded(
        enabled=True,
        strict=True,
    )
    if not native_status_object.loaded:
        raise RuntimeError("installed TriangleMix native bootstrap failed")

    benchmark_config = kernel.TriangleMixConfig(
        enabled=True,
        layer_indices=frozenset({0}),
        direct_min_seq_len=0,
        direct_min_sparse_rows=0,
        direct_min_saved_qk=0,
        direct_split_min_sparse_rows=0,
        direct_split_min_saved_qk=0,
    )
    production_config = kernel.TriangleMixConfig(
        enabled=True,
        layer_indices=frozenset({0}),
    )
    cases = build_route_cases(
        lengths=lengths,
        chunk_sizes=chunk_sizes,
        seed=args.seed,
        wrapper_case_type=WrapperCase,
    )
    lock_path = (
        args.lock_path.resolve()
        if args.lock_path is not None
        else DEFAULT_NPU_LOCK
    )
    records: list[dict[str, Any]] = []
    preflights: list[dict[str, Any]] = []
    memory_before = None
    memory_after_inputs = None
    started = time.time()
    with exclusive_npu_lock(lock_path, args.lock_timeout):
        torch.npu.set_device(args.device)
        memory_before = memory_info()
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
                    wrapper=kernel,
                    config=benchmark_config,
                    case=case,
                    inputs=inputs,
                    dense_region_max_abs=args.dense_region_max_abs,
                    dense_region_mean_abs=args.dense_region_mean_abs,
                    sparse_stage_max_abs=args.sparse_stage_max_abs,
                )
                preflights.append({"name": case.name, **preflight})
                if preflight["status"] != "PASS":
                    raise AssertionError(
                        f"{case.name}: device preflight failed"
                    )
                record = benchmark_cell(
                    wrapper=kernel,
                    config=benchmark_config,
                    case=case,
                    inputs=inputs,
                    warmup=args.warmup,
                    iterations=args.iterations,
                    repeats=args.repeats,
                    cell_index=cell_index,
                )
                # The reused harness predates the release vocabulary. This is
                # an attention-only microbenchmark, never end-to-end TTFT.
                _normalize_attention_record(record)
                release_direct = _production_route(
                    kernel,
                    production_config,
                    case,
                )
                record["chunk_size"] = next(
                    chunk
                    for chunk in chunk_sizes
                    if f"_c{chunk}_" in case.name
                )
                record["release_routed_direct"] = release_direct
                record["release_false_positive_direct"] = (
                    release_direct
                    and not bool(record["robust_direct_faster"])
                )
                record["abba_order_valid"] = _is_exact_abba(
                    record.get("repeat_orders")
                )
                if not record["abba_order_valid"]:
                    raise AssertionError(
                        f"{case.name}: invalid ABBA order"
                    )
            except Exception as error:
                record = {
                    **case.to_dict(),
                    "chunk_size": next(
                        chunk
                        for chunk in chunk_sizes
                        if f"_c{chunk}_" in case.name
                    ),
                    "status": "error",
                    "plan": plan_for_case(case).to_dict(),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            records.append(record)
            print(json.dumps(record, sort_keys=True), flush=True)
            if (
                record.get("status") != "ok"
                and not args.continue_on_error
            ):
                break
        torch.npu.synchronize()

    routes = summarize_routes(
        records,
        lengths=lengths,
        chunk_sizes=chunk_sizes,
    )
    false_positives = [
        record["name"]
        for record in records
        if record.get("release_false_positive_direct")
    ]
    checks = {
        "correctness_prerequisite_exact_wheel": True,
        "all_cells_completed": len(records) == len(cases)
        and all(record.get("status") == "ok" for record in records),
        "all_event_orders_exact_abba": bool(records)
        and all(record.get("abba_order_valid") for record in records),
        "all_routes_complete": len(routes)
        == len(lengths) * len(chunk_sizes)
        and all(route["complete"] for route in routes),
        "release_false_positive_direct_zero": not false_positives,
        "batch_size_is_one": True,
    }
    passed = all(checks.values())
    report = {
        "schema_version": 1,
        "suite": "trianglemix_installed_wheel_npu_crossover",
        "status": "PASS" if passed else "FAIL",
        "claim": "B=1 attention-only NPU Event crossover",
        "not_a_ttft_measurement": True,
        "attention_microbenchmark_used": True,
        "wheel": {
            "path": str(wheel_path),
            "sha256": wheel_sha256,
        },
        "correctness_prerequisite": prerequisite,
        "installed": provenance,
        "native_status": native_status_object.to_dict(),
        "runtime": {
            **environment_fingerprint(),
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "device": torch.npu.get_device_name(args.device),
            "platform": platform.platform(),
        },
        "measurement": {
            "clock": "torch.npu.Event",
            "batch_size": 1,
            "warmup": args.warmup,
            "iterations_per_sample": args.iterations,
            "repeats": args.repeats,
            "order": "paired AB/BA; every adjacent pair is exact ABBA",
            "baseline": "official dense paged FIA",
            "candidate": "installed wheel single-launch direct kernel",
        },
        "matrix": {
            "lengths": lengths,
            "chunk_sizes": chunk_sizes,
            "cells": len(cases),
        },
        "production_thresholds": {
            "min_seq_len": production_config.direct_min_seq_len,
            "min_sparse_rows": production_config.direct_min_sparse_rows,
            "min_saved_qk": production_config.direct_min_saved_qk,
            "split_min_sparse_rows": (
                production_config.direct_split_min_sparse_rows
            ),
            "split_min_saved_qk": (
                production_config.direct_split_min_saved_qk
            ),
        },
        "checks": checks,
        "release_false_positive_direct_cells": false_positives,
        "routes": routes,
        "preflight": preflights,
        "records": records,
        "lock_path": str(lock_path),
        "memory": {
            "before": memory_before,
            "after_inputs": memory_after_inputs,
        },
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
    }
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cells": len(records),
                "false_positive_direct": len(false_positives),
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
