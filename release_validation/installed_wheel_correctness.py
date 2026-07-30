#!/usr/bin/env python3
"""Run the NPU correctness oracle through an installed TriangleMix wheel.

Unlike the development harness, this command never loads a wrapper source file
or accepts an adapter path.  Native bootstrap and the candidate call both come
from the installed ``vllm_ascend_trianglemix`` distribution.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

from .common import (
    environment_fingerprint,
    is_relative_to,
    sha256_file,
    write_json,
)
from .installed_artifact import audit_installed_wheel


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = PROJECT_ROOT / "harness"


def _select_cases(all_cases: tuple[Any, ...], names: list[str]) -> tuple[Any, ...]:
    if not names:
        return all_cases
    by_name = {str(case.name): case for case in all_cases}
    unknown = sorted(set(names) - set(by_name))
    if unknown:
        raise ValueError(f"unknown correctness cases: {', '.join(unknown)}")
    return tuple(by_name[name] for name in names)


def _installed_provenance(
    kernel_module: object,
    native_module: object,
    wheel_path: Path,
) -> dict[str, Any]:
    distribution = metadata.distribution("vllm-ascend-trianglemix")
    distribution_root = Path(distribution.locate_file("")).resolve()
    kernel_path = Path(str(getattr(kernel_module, "__file__"))).resolve()
    native_path = Path(str(getattr(native_module, "__file__"))).resolve()
    installed = (
        is_relative_to(kernel_path, distribution_root)
        and is_relative_to(native_path, distribution_root)
        and "/package/src/" not in kernel_path.as_posix()
        and "/package/src/" not in native_path.as_posix()
    )
    artifact_audit = audit_installed_wheel(
        wheel_path,
        distribution=distribution,
    )
    return {
        "distribution_name": distribution.metadata.get("Name"),
        "distribution_version": distribution.version,
        "distribution_root": str(distribution_root),
        "kernel_module": str(kernel_path),
        "kernel_sha256": sha256_file(kernel_path),
        "native_module": str(native_path),
        "native_sha256": sha256_file(native_path),
        "is_installed_non_source_copy": installed,
        "exact_wheel_payload_match": artifact_audit,
    }


def _config(kernel: object) -> object:
    return kernel.TriangleMixConfig(
        enabled=True,
        layer_indices=frozenset({0}),
        sink_tokens=8,
        local_window=512,
        last_rows=128,
        direct_min_seq_len=0,
        direct_min_sparse_rows=0,
        direct_min_saved_qk=0,
        direct_split_min_sparse_rows=0,
        direct_split_min_saved_qk=0,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--wheel",
        type=Path,
        required=True,
        help="The exact wheel artifact already installed in this environment.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lock-path", type=Path)
    parser.add_argument("--lock-timeout", type=float, default=300.0)
    parser.add_argument("--dense-region-max-abs", type=float, default=0.04)
    parser.add_argument("--dense-region-mean-abs", type=float, default=0.004)
    parser.add_argument("--sparse-stage-max-abs", type=float, default=0.0)
    parser.add_argument(
        "--triangle-reference-max-abs",
        type=float,
        default=0.04,
    )
    parser.add_argument(
        "--triangle-reference-mean-abs",
        type=float,
        default=0.004,
    )
    parser.add_argument("--continue-on-error", action="store_true")
    args = parser.parse_args(argv)

    wheel_path = args.wheel.resolve()
    output_path = args.output.resolve()
    if not wheel_path.is_file():
        parser.error(f"wheel does not exist: {wheel_path}")
    if not HARNESS_DIR.is_dir():
        parser.error(f"harness directory does not exist: {HARNESS_DIR}")
    if args.lock_timeout < 0:
        parser.error("--lock-timeout must be non-negative")
    for name in (
        "dense_region_max_abs",
        "dense_region_mean_abs",
        "sparse_stage_max_abs",
        "triangle_reference_max_abs",
        "triangle_reference_mean_abs",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be non-negative")

    # This exposes only the independent harness.  In particular, package/src
    # is never added to sys.path, so the candidate imports below must resolve
    # from the installed distribution.
    sys.path.insert(0, str(HARNESS_DIR))
    import torch
    import torch_npu
    from harness_spec import CORRECTNESS_CASES
    from npu_harness_common import (
        DEFAULT_NPU_LOCK,
        evaluate_wrapper_case,
        exclusive_npu_lock,
        make_npu_inputs,
        memory_info,
    )

    cases = _select_cases(CORRECTNESS_CASES, args.case)
    lock_path = (
        args.lock_path.resolve()
        if args.lock_path is not None
        else DEFAULT_NPU_LOCK
    )
    records: list[dict[str, Any]] = []
    error: dict[str, str] | None = None
    memory_before = None
    memory_after = None
    started = time.time()
    provenance: dict[str, Any] = {}
    native_status: dict[str, Any] | None = None

    try:
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
        required_kernel_symbols = (
            "TriangleMixConfig",
            "triangle_direct_eligible",
            "triangle_direct_paged_attention",
            "triangle_sparse_span",
        )
        missing_symbols = [
            name
            for name in required_kernel_symbols
            if not callable(getattr(kernel, name, None))
        ]
        if missing_symbols:
            raise RuntimeError(
                "installed kernel is missing required symbols: "
                + ", ".join(missing_symbols)
            )
        status = native.ensure_native_loaded(enabled=True, strict=True)
        native_status = status.to_dict()
        if not status.loaded:
            raise RuntimeError(
                f"installed native bootstrap did not load: {native_status}"
            )
        config = _config(kernel)

        with exclusive_npu_lock(lock_path, args.lock_timeout):
            torch.npu.set_device(args.device)
            memory_before = memory_info()
            for case in cases:
                try:
                    inputs = make_npu_inputs(
                        maximum_seq_len=case.seq_len,
                        maximum_query_tokens=case.query_tokens,
                        seed=case.seed,
                    )
                    torch.npu.synchronize()
                    record = evaluate_wrapper_case(
                        wrapper=kernel,
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
                except Exception as case_error:
                    record = {
                        **case.to_dict(),
                        "status": "ERROR",
                        "error_type": type(case_error).__name__,
                        "error": str(case_error),
                        "traceback": traceback.format_exc(),
                    }
                records.append(record)
                print(json.dumps(record, sort_keys=True), flush=True)
                if (
                    record.get("status") != "PASS"
                    and not args.continue_on_error
                ):
                    break
            torch.npu.synchronize()
            memory_after = memory_info()
    except Exception as caught:
        error = {
            "type": type(caught).__name__,
            "message": str(caught),
            "traceback": traceback.format_exc(),
        }

    passed = (
        error is None
        and len(records) == len(cases)
        and all(record.get("status") == "PASS" for record in records)
    )
    report = {
        "schema_version": 1,
        "suite": "trianglemix_installed_wheel_npu_correctness",
        "status": "PASS" if passed else "FAIL",
        "claim": (
            "NPU operator correctness through installed "
            "vllm_ascend_trianglemix.native and .kernel"
        ),
        "not_a_ttft_measurement": True,
        "wheel": {
            "path": str(wheel_path),
            "sha256": sha256_file(wheel_path),
        },
        "installed": provenance,
        "native_status": native_status,
        "runtime": {
            **environment_fingerprint(),
            "torch": str(torch.__version__),
            "torch_npu": str(torch_npu.__version__),
            "device": (
                torch.npu.get_device_name(args.device)
                if error is None or records
                else None
            ),
            "platform": platform.platform(),
        },
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
        "memory": {
            "before": memory_before,
            "after": memory_after,
        },
        "lock_path": str(lock_path),
        "selected_cases": [case.to_dict() for case in cases],
        "records": records,
        "error": error,
        "started_unix_seconds": started,
        "finished_unix_seconds": time.time(),
    }
    write_json(output_path, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "cases": len(records),
                "output": str(output_path),
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
