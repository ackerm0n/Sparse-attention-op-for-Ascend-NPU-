#!/usr/bin/env python3
"""Validate caller-owned output, storage offsets, and guard preservation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from npu_v2_correctness import CorrectnessCase, make_random_inputs
from npu_v2_harness_common import DEFAULT_NPU_LOCK, exclusive_npu_lock


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--lock-path", type=Path, default=DEFAULT_NPU_LOCK)
    parser.add_argument("--lock-timeout", type=float, default=120.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    adapter = args.adapter.resolve()
    if not adapter.is_file():
        raise FileNotFoundError(adapter)

    case = CorrectnessCase(
        "out_nonzero_storage_offset",
        query_start=766,
        query_tokens=35,
        seq_len=801,
        prompt_len=2048,
        seed=3101,
        coverage=(
            "ragged sparse query, non-contiguous pages, caller output "
            "with prefix and suffix guards"
        ),
    )
    query, key, value, block_table = make_random_inputs(case)
    scale = query.shape[-1] ** -0.5

    with exclusive_npu_lock(args.lock_path, args.lock_timeout):
        torch.npu.set_device(args.device)
        torch.ops.load_library(str(adapter))
        query_npu = query.npu()
        key_npu = key.npu()
        value_npu = value.npu()
        table_npu = block_table.npu()

        functional = (
            torch.ops.trianglemix_reference
            .triangle_paged_sparse_attention(
                query_npu,
                key_npu,
                value_npu,
                table_npu,
                case.query_start,
                case.seq_len,
                case.prompt_len,
                scale,
            )
        )

        guarded = torch.full(
            (
                case.query_tokens + 2,
                query.shape[1],
                query.shape[2],
            ),
            17.0,
            dtype=query.dtype,
            device=query_npu.device,
        )
        output = guarded[1:-1]
        output.fill_(-31.0)
        pointer_before = output.data_ptr()
        storage_offset_before = output.storage_offset()
        returned = (
            torch.ops.trianglemix
            .triangle_paged_sparse_attention.out(
                query_npu,
                key_npu,
                value_npu,
                table_npu,
                case.query_start,
                case.seq_len,
                case.prompt_len,
                scale,
                out=output,
            )
        )
        torch.npu.synchronize()

        maximum_error = float(
            (output.float() - functional.float()).abs().max().cpu()
        )
        prefix_ok = bool(torch.all(guarded[0] == 17.0).cpu())
        suffix_ok = bool(torch.all(guarded[-1] == 17.0).cpu())
        alias_rejected = False
        try:
            torch.ops.trianglemix.triangle_paged_sparse_attention.out(
                query_npu,
                key_npu,
                value_npu,
                table_npu,
                case.query_start,
                case.seq_len,
                case.prompt_len,
                scale,
                out=query_npu,
            )
        except RuntimeError:
            alias_rejected = True
        report = {
            "status": "PASS",
            "case": case.name,
            "max_abs_vs_functional": maximum_error,
            "output_contiguous": output.is_contiguous(),
            "storage_offset": storage_offset_before,
            "same_data_ptr": returned.data_ptr() == pointer_before,
            "same_storage_offset": (
                returned.storage_offset() == storage_offset_before
            ),
            "prefix_guard_preserved": prefix_ok,
            "suffix_guard_preserved": suffix_ok,
            "query_alias_rejected": alias_rejected,
        }
        required = (
            maximum_error == 0.0
            and report["output_contiguous"]
            and storage_offset_before > 0
            and report["same_data_ptr"]
            and report["same_storage_offset"]
            and prefix_ok
            and suffix_ok
            and alias_rejected
        )
        if not required:
            report["status"] = "FAIL"

    print(json.dumps(report, ensure_ascii=False), flush=True)
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
