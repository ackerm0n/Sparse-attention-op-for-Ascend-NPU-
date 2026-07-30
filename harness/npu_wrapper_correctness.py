#!/usr/bin/env python3
"""Validate the real direct wrapper against one-shot official paged FIA.

Sparse middle rows intentionally implement TriangleMix rather than dense
attention, so the full-output dense delta is reported as a quality metric.
Correctness gates are:

* no-middle cases take one official dense fallback and match dense output;
* dense prefix/tail slices match the corresponding rows from one dense call;
* the middle matches an independent CPU FP32 sink+local Triangle reference;
* the middle also exactly matches a standalone final adapter ``.out`` call;
* the wrapper returns the caller-owned output object and all outputs are finite.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import torch
import torch_npu

from harness_spec import CORRECTNESS_CASES, WrapperCase
from npu_harness_common import (
    DEFAULT_NPU_LOCK,
    evaluate_wrapper_case,
    exclusive_npu_lock,
    file_sha256,
    load_candidate_module,
    make_npu_inputs,
    memory_info,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wrapper",
        type=Path,
        required=True,
        help="Candidate triangle_flash_attention.py",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        required=True,
        help="Final Torch adapter shared library with trianglemix .out",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        help="Run only this named case; may be repeated",
    )
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
        help="Wrapper middle vs the same final adapter called standalone",
    )
    parser.add_argument(
        "--triangle-reference-max-abs",
        type=float,
        default=0.04,
        help="Sparse middle vs independent CPU FP32 Triangle reference",
    )
    parser.add_argument(
        "--triangle-reference-mean-abs",
        type=float,
        default=0.004,
    )
    parser.add_argument(
        "--lock-path",
        type=Path,
        default=DEFAULT_NPU_LOCK,
    )
    parser.add_argument("--lock-timeout", type=float, default=300.0)
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue after a failed/error case (unsafe after device errors)",
    )
    return parser.parse_args()


def select_cases(names: list[str] | None) -> tuple[WrapperCase, ...]:
    if not names:
        return CORRECTNESS_CASES
    by_name = {case.name: case for case in CORRECTNESS_CASES}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown cases: {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "dense_region_max_abs",
        "dense_region_mean_abs",
        "sparse_stage_max_abs",
        "triangle_reference_max_abs",
        "triangle_reference_mean_abs",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.lock_timeout < 0:
        raise ValueError("lock timeout must be non-negative")


def main() -> int:
    args = parse_args()
    validate_args(args)
    cases = select_cases(args.case)
    wrapper_path = args.wrapper.resolve()
    adapter_path = args.adapter.resolve()
    output_path = args.output.resolve()
    if not wrapper_path.is_file():
        raise FileNotFoundError(wrapper_path)
    if not adapter_path.is_file():
        raise FileNotFoundError(adapter_path)

    records: list[dict[str, Any]] = []
    memory_before = None
    memory_after = None
    started = time.time()
    with exclusive_npu_lock(args.lock_path, args.lock_timeout):
        torch.npu.set_device(args.device)
        memory_before = memory_info()
        wrapper, config = load_candidate_module(
            wrapper_path,
            adapter_path,
        )
        for case in cases:
            try:
                inputs = make_npu_inputs(
                    maximum_seq_len=case.seq_len,
                    maximum_query_tokens=case.query_tokens,
                    seed=case.seed,
                )
                torch.npu.synchronize()
                record = evaluate_wrapper_case(
                    wrapper=wrapper,
                    config=config,
                    case=case,
                    inputs=inputs,
                    dense_region_max_abs=args.dense_region_max_abs,
                    dense_region_mean_abs=args.dense_region_mean_abs,
                    sparse_stage_max_abs=args.sparse_stage_max_abs,
                    triangle_reference_max_abs=(
                        args.triangle_reference_max_abs
                    ),
                    triangle_reference_mean_abs=(
                        args.triangle_reference_mean_abs
                    ),
                )
                del inputs
            except Exception as error:
                record = {
                    **case.to_dict(),
                    "status": "ERROR",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                }
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
            if (
                record["status"] != "PASS"
                and not args.continue_on_error
            ):
                break
        torch.npu.synchronize()
        memory_after = memory_info()

    all_passed = (
        len(records) == len(cases)
        and all(record["status"] == "PASS" for record in records)
    )
    script_path = Path(__file__).resolve()
    common_path = script_path.with_name("npu_harness_common.py")
    spec_path = script_path.with_name("harness_spec.py")
    report = {
        "schema_version": 1,
        "suite": "trianglemix_direct_wrapper_correctness",
        "status": "PASS" if all_passed else "FAIL",
        "geometry": {
            "dtype": "bfloat16",
            "query_heads": 32,
            "kv_heads": 8,
            "head_dim": 128,
            "page_size": 128,
            "sink_tokens": 8,
            "local_window": 512,
            "dense_tail": 128,
            "batch": 1,
        },
        "comparison": {
            "baseline": (
                "single torch.ops.npu."
                "npu_fused_infer_attention_score.out paged FIA"
            ),
            "candidate": (
                "loaded candidate triangle_direct_paged_attention with "
                "final trianglemix .out adapter"
            ),
            "full_candidate_vs_dense": (
                "reported quality delta; sparse rows are intentionally "
                "not required to equal dense attention"
            ),
            "dense_region_gate": {
                "max_abs": args.dense_region_max_abs,
                "mean_abs": args.dense_region_mean_abs,
            },
            "middle_standalone_gate_max_abs": (
                args.sparse_stage_max_abs
            ),
            "middle_cpu_fp32_triangle_gate": {
                "max_abs": args.triangle_reference_max_abs,
                "mean_abs": args.triangle_reference_mean_abs,
                "schedule": (
                    "joint softmax over sink [0,8) plus local "
                    "[q-512,q] inclusive"
                ),
            },
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
            "after": memory_after,
        },
        "selected_cases": [case.to_dict() for case in cases],
        "records": records,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"CORRECTNESS_{report['status']}", flush=True)
    print(f"result={output_path}", flush=True)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
